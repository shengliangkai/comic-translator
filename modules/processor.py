import os
import sys
import time
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from .ocr import ComicOCR
from .translator import Translator
from .remover import TextRemover
from .typesetter import Typesetter


@dataclass
class OCRResult:
    index: int = 0
    text: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    polygon: List[List[float]] = field(default_factory=list)
    confidence: float = 0.0
    translation: str = ""
    bg_color_rgb: Tuple[int, int, int] = (255, 255, 255)
    text_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    is_white_bg: bool = True
    user_edited: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ComicTranslator:
    def __init__(
        self,
        ocr_lang: str = None,
        translate_engine: str = None,
        inpaint_method: str = None,
        font_path: str = None,
        ocr_accuracy: str = None,
    ):
        self.ocr = ComicOCR(lang=ocr_lang, accuracy=ocr_accuracy)
        self.translator = Translator(engine=translate_engine)
        self.remover = TextRemover(method=inpaint_method)
        self.typesetter = Typesetter(font_path=font_path)
        self._last_results: List[OCRResult] = []

    def _to_ocr_results(
        self, raw_items: List[Dict[str, Any]]
    ) -> List[OCRResult]:
        results = []
        for i, item in enumerate(raw_items):
            r = OCRResult(
                index=i,
                text=item.get("text", ""),
                bbox=tuple(item.get("bbox", (0, 0, 0, 0))),
                polygon=item.get("polygon", []),
                confidence=float(item.get("confidence", 0.0)),
            )
            results.append(r)
        return results

    def do_ocr(
        self, image_path: Union[str, np.ndarray], expand_bbox: bool = True
    ) -> List[OCRResult]:
        print("[流程] 步骤 1/5: OCR 识别...")
        t0 = time.time()
        raw = self.ocr.recognize(image_path, expand_bbox=expand_bbox)
        results = self._to_ocr_results(raw)
        self._last_results = results
        print(f"[流程] 识别到 {len(results)} 个文本区域 (用时 {time.time()-t0:.2f}s)")
        for r in results:
            print(f"       #{r.index}: {r.text}  置信度={r.confidence:.2f}")
        return results

    def do_translate(
        self,
        results: List[OCRResult] = None,
        source_lang: str = "auto",
        target_lang: str = "zh",
    ) -> List[OCRResult]:
        print("[流程] 步骤 2/5: 文本翻译...")
        t0 = time.time()
        results = results or self._last_results
        if not results:
            return []

        texts = [r.text for r in results]
        translated = self.translator.translate_batch(texts, source_lang, target_lang)

        for r, tr in zip(results, translated):
            r.translation = tr
            print(f"       #{r.index}: {r.text}  →  {r.translation}")

        print(f"[流程] 翻译完成 (用时 {time.time()-t0:.2f}s)")
        return results

    def do_remove(
        self,
        image: Union[str, np.ndarray],
        results: List[OCRResult] = None,
    ) -> Tuple[np.ndarray, List[OCRResult]]:
        print("[流程] 步骤 3/5: 清除原文...")
        t0 = time.time()
        results = results or self._last_results
        boxes = [r.bbox for r in results]

        cleaned_img, bubble_info = self.remover.remove_text(
            image, boxes, return_colors=True
        )

        info_map = {tuple(bi["bbox"]): bi for bi in bubble_info}
        for r in results:
            info = info_map.get(tuple(r.bbox))
            if info:
                r.bg_color_rgb = tuple(info["bg_color_rgb"])
                r.text_color_rgb = tuple(info.get("text_color_rgb", (0, 0, 0)))
                r.is_white_bg = bool(info.get("is_white_bg", True))

        print(f"[流程] 原文清除完成 (用时 {time.time()-t0:.2f}s)")
        return cleaned_img, results

    def do_typeset(
        self,
        cleaned_image: np.ndarray,
        results: List[OCRResult] = None,
        use_user_edits: bool = True,
    ) -> np.ndarray:
        print("[流程] 步骤 4/5: 中文嵌字排版...")
        t0 = time.time()
        results = results or self._last_results

        items = []
        for r in results:
            items.append(
                {
                    "bbox": r.bbox,
                    "text": r.text,
                    "translation": r.translation,
                    "text_color_rgb": r.text_color_rgb,
                    "bg_color_rgb": None if r.is_white_bg else r.bg_color_rgb,
                }
            )

        final_img = self.typesetter.add_text(cleaned_image, items)
        print(f"[流程] 嵌字完成 (用时 {time.time()-t0:.2f}s)")
        return final_img

    def process(
        self,
        image_path: Union[str, np.ndarray],
        source_lang: str = "auto",
        target_lang: str = "zh",
        output_path: str = None,
        return_steps: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        print("=" * 50)
        print(f"[流程] 开始处理: {image_path if isinstance(image_path, str) else '<numpy array>'}")
        total_t0 = time.time()

        results = self.do_ocr(image_path)
        results = self.do_translate(results, source_lang, target_lang)

        preview = None
        if isinstance(image_path, str):
            preview = self.typesetter.draw_bboxes(image_path, [r.to_dict() for r in results])

        cleaned_img, results = self.do_remove(image_path, results)

        final_img = self.do_typeset(cleaned_img, results)

        print(f"[流程] 全部完成，总用时 {time.time()-total_t0:.2f}s")
        print("=" * 50)

        if output_path:
            self.typesetter.save_image(final_img, output_path)
            print(f"[流程] 输出已保存: {output_path}")

        if return_steps:
            return {
                "final_image": final_img,
                "cleaned_image": cleaned_img,
                "preview_image": preview,
                "results": [r.to_dict() for r in results],
            }
        return final_img

    def get_results(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._last_results]

    def update_result(
        self, index: int, translation: str = None, text: str = None
    ) -> bool:
        if 0 <= index < len(self._last_results):
            r = self._last_results[index]
            if translation is not None:
                r.translation = translation
                r.user_edited = True
            if text is not None:
                r.text = text
                r.user_edited = True
            return True
        return False

    def save_results_json(self, results: List[OCRResult], output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data = [r.to_dict() for r in results]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[流程 错误] 保存 JSON 失败: {e}")
            return False
