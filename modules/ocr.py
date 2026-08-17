import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ComicOCR:
    """OCR wrapper that supports PaddleOCR 2.x and 3.x, with Mock fallback.

    - PaddleOCR 3.x: uses ``predict()`` and the modern ``PaddleOCR(...)`` kwargs
      (``use_textline_orientation``, ``use_doc_orientation_classify``, ...).
    - PaddleOCR 2.x: uses the legacy ``.ocr(path, cls=False)`` API.
    - No PaddleOCR installed or init fails: Mock mode with fixed demo regions.

    Manga-specific tuning: text-line orientation is enabled so vertical Japanese
    dialogue is read correctly, and the detection side limit is raised so small
    speech-bubble text is not downscaled out of readability.
    """

    def __init__(self, lang: str = None, use_gpu: bool = None, accuracy: str = None):
        self.lang = lang or config.OCR_LANG
        self.use_gpu = use_gpu if use_gpu is not None else config.OCR_USE_GPU
        self.accuracy = accuracy or config.OCR_ACCURACY
        self._ocr = None
        self._is_mock = True
        self._init_ocr()

    @property
    def is_mock(self) -> bool:
        """True when OCR is running in Mock mode (no real recognition)."""
        return self._is_mock

    def _init_ocr(self):
        # 离线优先：优先使用项目内置的模型缓存，避免联网检查和用户目录权限问题。
        local_cache = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "paddlex",
        )
        if os.path.isdir(os.path.join(local_cache, "official_models")):
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", local_cache)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # Disable oneDNN/MKLDNN: it hits an unimplemented-op error in
        # PaddlePaddle 3.x on Windows for some OCR ops.
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        self.init_error = None
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            self.init_error = str(e)
            print(f"[OCR 警告] PaddleOCR 未安装: {e}")
            print("[OCR 提示] 将使用 Mock OCR 模式，仅用于测试流程")
            self._ocr = None
            self._is_mock = True
            return

        major = self._paddleocr_major_version()
        if major >= 3:
            self._init_paddle3(PaddleOCR)
        else:
            self._init_paddle2(PaddleOCR)

        if self._ocr is None:
            print("[OCR 提示] 将使用 Mock OCR 模式，仅用于测试流程")
            self._is_mock = True
        else:
            self._is_mock = False

    @staticmethod
    def _paddleocr_major_version() -> int:
        try:
            import paddleocr

            ver = getattr(paddleocr, "__version__", "2")
            return int(str(ver).split(".")[0])
        except Exception:
            return 2

    def _init_paddle3(self, PaddleOCR):
        """PaddleOCR 3.x: modern kwargs, predict()-based API."""
        params = dict(
            lang=self.lang,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            # 大图小字：提高检测边长上限，减少对原始分辨率的压缩
            text_det_limit_type="max",
            text_det_limit_side_len=1920,
            text_det_box_thresh=0.35,
            text_det_unclip_ratio=1.8,
        )
        if self.accuracy == "high":
            # 高精度：服务器级检测（小字框得更稳） + v6 中等识别（日文假名更准）
            params["text_detection_model_name"] = "PP-OCRv5_server_det"
            params["text_recognition_model_name"] = "PP-OCRv6_medium_rec"
        try:
            self._ocr = PaddleOCR(**params)
        except Exception as e:
            print(f"[OCR 警告] PaddleOCR 3.x 参数不受支持，回退重试: {e}")
            self.init_error = str(e)
            params.pop("text_detection_model_name", None)
            params.pop("text_recognition_model_name", None)
            params.pop("text_det_limit_side_len", None)
            params.pop("text_det_box_thresh", None)
            params.pop("text_det_unclip_ratio", None)
            try:
                self._ocr = PaddleOCR(**params)
            except Exception as e2:
                print(f"[OCR 警告] PaddleOCR 3.x 初始化失败: {e2}")
                self.init_error = str(e2)
                try:
                    self._ocr = PaddleOCR(lang=self.lang)
                except Exception as e3:
                    print(f"[OCR 警告] PaddleOCR 3.x 初始化失败: {e3}")
                    self.init_error = str(e3)
                    self._ocr = None

    def _init_paddle2(self, PaddleOCR):
        """PaddleOCR 2.x: legacy kwargs, .ocr()-based API."""
        params = dict(
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            det_limit_side_len=1280,
        )
        try:
            self._ocr = PaddleOCR(**params)
        except TypeError:
            params.pop("use_doc_orientation_classify", None)
            params.pop("use_doc_unwarping", None)
            params.pop("use_textline_orientation", None)
            try:
                self._ocr = PaddleOCR(**params)
            except TypeError:
                try:
                    self._ocr = PaddleOCR(lang=self.lang, show_log=False)
                except Exception as e:
                    print(f"[OCR 警告] PaddleOCR 初始化失败: {e}")
                    self.init_error = str(e)
                    self._ocr = None
        except Exception as e:
            print(f"[OCR 警告] PaddleOCR 初始化失败: {e}")
            self.init_error = str(e)
            self._ocr = None

    def _polygon_to_bbox(self, polygon) -> Tuple[int, int, int, int]:
        pts = np.asarray(polygon, dtype=float)
        if pts.size == 0:
            return (0, 0, 0, 0)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        return (x1, y1, x2, y2)

    def _expand_bbox(
        self, bbox: Tuple[int, int, int, int], padding: int = None
    ) -> Tuple[int, int, int, int]:
        pad = padding if padding is not None else config.BBOX_PADDING
        x1, y1, x2, y2 = bbox
        return (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)

    def _mock_ocr(self, image_path: str) -> List[Dict[str, Any]]:
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        return [
            {
                "text": "こんにちは！",
                "bbox": self._expand_bbox((int(w * 0.1), int(h * 0.05), int(w * 0.45), int(h * 0.18))),
                "polygon": [
                    [w * 0.1, h * 0.05],
                    [w * 0.45, h * 0.05],
                    [w * 0.45, h * 0.18],
                    [w * 0.1, h * 0.18],
                ],
                "confidence": 0.95,
            },
            {
                "text": "何をしているの？",
                "bbox": self._expand_bbox((int(w * 0.55), int(h * 0.55), int(w * 0.9), int(h * 0.7))),
                "polygon": [
                    [w * 0.55, h * 0.55],
                    [w * 0.9, h * 0.55],
                    [w * 0.9, h * 0.7],
                    [w * 0.55, h * 0.7],
                ],
                "confidence": 0.93,
            },
            {
                "text": "ありがとう",
                "bbox": self._expand_bbox((int(w * 0.3), int(h * 0.8), int(w * 0.55), int(h * 0.9))),
                "polygon": [
                    [w * 0.3, h * 0.8],
                    [w * 0.55, h * 0.8],
                    [w * 0.55, h * 0.9],
                    [w * 0.3, h * 0.9],
                ],
                "confidence": 0.97,
            },
        ]

    def _recognize_paddle3(
        self, image_path: str, expand_bbox: bool = True
    ) -> List[Dict[str, Any]]:
        """Parse PaddleOCR 3.x ``predict()`` output."""
        try:
            result = self._ocr.predict(image_path)
        except Exception as e:
            print(f"[OCR 错误] 识别失败: {e}")
            return self._mock_ocr(image_path)

        texts = []
        for page in result:
            if hasattr(page, "json"):
                j = page.json
                if callable(j):
                    j = j()
            else:
                try:
                    j = dict(page)
                except Exception:
                    continue
            # PaddleOCR 3.x nests the actual result under the "res" key.
            res = j.get("res") if isinstance(j, dict) else None
            if not isinstance(res, dict):
                res = j

            rec_texts = res.get("rec_texts") or []
            rec_scores = res.get("rec_scores") or []
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("dt_polys")
            if polys is None:
                continue

            for i, text in enumerate(rec_texts):
                if not text or not str(text).strip():
                    continue
                score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                if score < 0.5:
                    continue
                # Skip single Latin letters/digits (usually sound effects).
                t = str(text).strip()
                if len(t) == 1 and t.isascii():
                    continue
                # Short pure-ASCII strings with low confidence: usually
                # on-page noise / sound effects (e.g. "88", "SB", "NO").
                if len(t) <= 3 and t.isascii() and score < 0.85:
                    continue
                if i >= len(polys):
                    break
                bbox = self._polygon_to_bbox(polys[i])
                if expand_bbox:
                    bbox = self._expand_bbox(bbox)
                texts.append(
                    {
                        "text": t,
                        "bbox": bbox,
                        "polygon": np.asarray(polys[i], dtype=float).tolist(),
                        "confidence": score,
                    }
                )

        texts.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        return texts

    def _recognize_paddle2(
        self, image_path: str, expand_bbox: bool = True
    ) -> List[Dict[str, Any]]:
        """Parse PaddleOCR 2.x ``.ocr(path, cls=False)`` output."""
        try:
            result = self._ocr.ocr(image_path, cls=False)
        except Exception as e:
            print(f"[OCR 错误] 识别失败: {e}")
            return self._mock_ocr(image_path)

        texts = []
        if not result:
            return texts

        for page in result:
            if not page:
                continue
            for item in page:
                try:
                    if len(item) < 2:
                        continue
                    polygon = item[0]
                    text_info = item[1]
                    text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                    confidence = (
                        text_info[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) > 1
                        else 0.0
                    )
                    if not text or not text.strip():
                        continue
                    bbox = self._polygon_to_bbox(polygon)
                    if expand_bbox:
                        bbox = self._expand_bbox(bbox)
                    texts.append(
                        {
                            "text": text.strip(),
                            "bbox": bbox,
                            "polygon": polygon,
                            "confidence": float(confidence),
                        }
                    )
                except Exception as e:
                    print(f"[OCR 警告] 解析单条结果失败: {e}")
                    continue

        texts.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        return texts

    def recognize(
        self, image_path: Union[str, np.ndarray], expand_bbox: bool = True
    ) -> List[Dict[str, Any]]:
        if self._ocr is None:
            return self._mock_ocr(image_path if isinstance(image_path, str) else "mock.jpg")

        if hasattr(self._ocr, "predict"):
            return self._recognize_paddle3(image_path, expand_bbox)
        return self._recognize_paddle2(image_path, expand_bbox)
