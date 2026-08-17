import os
import sys
import time
import uuid
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import gradio as gr
from PIL import Image

from modules.ocr import ComicOCR
from modules.translator import Translator
from modules.remover import TextRemover
from modules.typesetter import Typesetter
from modules.processor import ComicTranslator


_CACHE: Dict[str, Dict[str, Any]] = {}


def _to_pil(img) -> Image.Image:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            return Image.fromarray(img)
        if len(img.shape) == 3 and img.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        if len(img.shape) == 3 and img.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return Image.fromarray(img)
    try:
        return Image.open(img).convert("RGB") if isinstance(img, str) else None
    except Exception:
        return None


def _to_cv2_bgr(img) -> np.ndarray:
    if img is None:
        raise ValueError("图片为空，请重新上传")
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if len(arr.shape) == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if len(arr.shape) == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            mean_vals = arr[:10, :10].mean()
            if mean_vals < 1 or mean_vals > 254:
                return arr
            return arr
        raise ValueError(f"不支持的图片格式 shape={arr.shape}")
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(img, str) and os.path.exists(img):
        arr = cv2.imread(img, cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"无法读取图片文件: {img}")
        return arr
    raise ValueError(f"无法识别的图片类型: {type(img)}")


def _save_uploaded_image(image) -> str:
    fname = f"{uuid.uuid4().hex[:12]}.png"
    fpath = os.path.join(config.TEMP_DIR, fname)
    img_bgr = _to_cv2_bgr(image)
    ok = cv2.imwrite(fpath, img_bgr)
    if not ok:
        raise RuntimeError(f"保存图片失败: {fpath}")
    return fpath


def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if len(img.shape) == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def build_editor_rows(results: List[Dict[str, Any]]) -> List[List[Any]]:
    if not results:
        return []
    rows = []
    for r in results:
        idx = r.get("index", 0)
        src = r.get("text", "")
        tr = r.get("translation", "")
        conf = f"{r.get('confidence', 0):.0%}"
        bbox = r.get("bbox", (0, 0, 0, 0))
        bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
        rows.append([idx, conf, bbox_str, src, tr])
    return rows


def parse_editor_rows(rows: List[List[Any]]) -> List[Dict[str, Any]]:
    updates = []
    for row in rows:
        if len(row) >= 5:
            try:
                idx = int(row[0])
            except Exception:
                continue
            src = str(row[3]) if row[3] is not None else ""
            tr = str(row[4]) if row[4] is not None else ""
            updates.append({"index": idx, "text": src, "translation": tr})
    return updates


def run_ocr_and_translate(
    image_pil,
    ocr_lang: str,
    translate_engine: str,
    source_lang: str,
) -> Tuple[Image.Image, List[List[Any]], str, str]:
    try:
        if image_pil is None:
            return None, [], "[错误] 请先上传漫画图片", ""

        try:
            image_path = _save_uploaded_image(image_pil)
        except Exception as e:
            return None, [], f"[错误] 保存上传图片失败: {e}", ""

        session_id = os.path.basename(image_path)

        processor = ComicTranslator(
            ocr_lang=ocr_lang,
            translate_engine=translate_engine,
        )

        steps = processor.process(image_path, source_lang=source_lang, return_steps=True)
        _CACHE[session_id] = {
            "processor": processor,
            "image_path": image_path,
            "steps": steps,
        }

        preview_pil = _cv2_to_pil(steps.get("preview_image"))
        results = steps.get("results", [])
        editor_rows = build_editor_rows(results)

        status_lines = []
        status_lines.append(f"[成功] OCR 识别完成，共检测到 {len(results)} 个文本区域")
        if translate_engine == "mock" and (not results or any(r.get("translation", "").startswith("[译文]") for r in results)):
            status_lines.append("[提示] 当前使用 Mock 翻译引擎，只能翻译内置示例句子。")
            status_lines.append("       真实翻译请设置环境变量后切换到 DeepSeek / OpenAI 引擎。")
          for r in results:
              status_lines.append(f"  #{r.get('index')}: {r.get('text')}  ->  {r.get('translation')}")
         if processor.ocr.is_mock:
             status_lines.append("[警告] 当前为 Mock 演示模式：未检测到可用的 PaddleOCR，红框与识别文字均为示例数据，不代表真实识别！")
  
          return preview_pil, editor_rows, "\n".join(status_lines), session_id
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        return None, [], f"[错误] 识别翻译失败: {e}\n{tb}", ""


def apply_edits_and_generate(
    image_pil,
    session_id: str,
    editor_rows: List[List[Any]],
    inpaint_method: str,
) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    try:
        if image_pil is None:
            return None, None, None, "[错误] 请先上传并识别漫画图片"

        if not session_id or session_id not in _CACHE:
            return None, None, None, "[错误] 会话已失效，请重新点击第一步识别"

        cache = _CACHE[session_id]
        processor: ComicTranslator = cache["processor"]
        image_path = cache["image_path"]
        steps = cache["steps"]

        updates = parse_editor_rows(editor_rows)
        for u in updates:
            idx = u["index"]
            if "translation" in u:
                processor.update_result(idx, translation=u["translation"])
            if "text" in u:
                processor.update_result(idx, text=u["text"])

        results = processor._last_results
        if not results:
            return None, None, None, "[错误] 没有识别到任何文本区域，请先运行第一步"

        remover = TextRemover(method=inpaint_method)
        boxes = [r.bbox for r in results]
        try:
            cleaned_cv, bubble_info = remover.remove_text(image_path, boxes, return_colors=True)
        except Exception as e:
            return None, None, None, f"[错误] 原文擦除失败: {e}"

        info_map = {tuple(bi["bbox"]): bi for bi in bubble_info}
        for r in results:
            info = info_map.get(tuple(r.bbox))
            if info:
                r.bg_color_rgb = tuple(info["bg_color_rgb"])
                r.text_color_rgb = tuple(info.get("text_color_rgb", (0, 0, 0)))
                r.is_white_bg = bool(info.get("is_white_bg", True))

        try:
            final_cv = processor.do_typeset(cleaned_cv, results)
        except Exception as e:
            return None, None, None, f"[错误] 嵌字排版失败: {e}"

        preview_cv = steps.get("preview_image")
        preview_pil = _cv2_to_pil(preview_cv)
        cleaned_pil = _cv2_to_pil(cleaned_cv)
        final_pil = _cv2_to_pil(final_cv)

        out_name = f"translated_{os.path.splitext(session_id)[0]}.png"
        out_path = os.path.join(config.OUTPUT_DIR, out_name)
        try:
            cv2.imwrite(out_path, final_cv)
        except Exception as e:
            return preview_pil, cleaned_pil, final_pil, f"[警告] 生成完成，但保存失败: {e}"

        status = (
            f"[成功] 中文版漫画生成完成！\n"
            f"   原文区域数: {len(results)}\n"
            f"   擦除方法: {inpaint_method}\n"
            f"   已保存至: {out_path}"
        )
        return preview_pil, cleaned_pil, final_pil, status
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        return None, None, None, f"[错误] 生成中文版失败: {e}\n{tb}"


def save_output_image(final_pil: Image.Image, session_id: str) -> str:
    if final_pil is None:
        return ""
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
    out_name = f"comic_translated_{session_id}_{int(time.time())}.png"
    out_path = os.path.join(config.OUTPUT_DIR, out_name)
    final_pil.save(out_path)
    return out_path


def create_demo():
    with gr.Blocks(
        title="🎨 AI 漫画翻译器 - Comic Translator",
    ) as demo:
        gr.Markdown(
            """
            # 🎨 AI 漫画自动翻译器
            **上传漫画图片 → OCR 识别 → 翻译 → 擦除原文 → 自动嵌字 → 输出中文版**

            💡 **使用提示**：第一阶段未安装 PaddleOCR 时将使用 **Mock 模式**（内置示例对白），
            安装依赖后可识别真实漫画。翻译引擎支持 Mock / DeepSeek / OpenAI。
            """
        )

        session_id = gr.State("")
        processor_ref = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 识别与翻译设置")
                with gr.Group():
                    ocr_lang = gr.Dropdown(
                        label="OCR 识别语言",
                        choices=[
                            ("日语漫画 (japan)", "japan"),
                            ("英语漫画 (en)", "en"),
                            ("韩语漫画 (korean)", "korean"),
                            ("中文简体 (ch)", "ch"),
                        ],
                        value="japan",
                    )
                    source_lang = gr.Dropdown(
                        label="翻译源语言",
                        choices=[
                            ("自动检测", "auto"),
                            ("日语", "ja"),
                            ("英语", "en"),
                            ("韩语", "ko"),
                        ],
                        value="auto",
                    )
                    translate_engine = gr.Dropdown(
                        label="翻译引擎",
                        choices=[
                            ("🔧 Mock 示例（测试用）", "mock"),
                            ("🧠 DeepSeek API", "deepseek"),
                            ("🤖 OpenAI API", "openai"),
                        ],
                        value="mock",
                    )
                    api_key_hint = gr.Markdown(
                        "🔑 **API Key 设置**：设置环境变量 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 即可"
                    )

                gr.Markdown("### 🖼️ 上传漫画")
                image_input = gr.Image(
                    label="漫画图片（JPG / PNG / WEBP）",
                    type="pil",
                    height=380,
                    image_mode="RGB",
                )

                with gr.Row():
                    btn_ocr = gr.Button(
                        "🔍 第一步：识别文字并翻译",
                        variant="primary",
                        size="lg",
                    )
                    btn_generate = gr.Button(
                        "✨ 第二步：生成中文版漫画",
                        variant="primary",
                        size="lg",
                    )

                gr.Markdown("### 🧹 擦除设置")
                inpaint_method = gr.Dropdown(
                    label="原文擦除方法",
                    choices=[
                        ("Telea 算法（较快）", "telea"),
                        ("Navier-Stokes（较平滑）", "ns"),
                    ],
                    value="telea",
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🔍 识别预览"):
                        image_preview = gr.Image(
                            label="OCR 检测结果（红框为识别到的文字区域）",
                            type="pil",
                            height=420,
                        )

                    with gr.Tab("📝 人工校对（重要！）"):
                        gr.Markdown(
                            "在这里可以**修改 OCR 原文**和**AI 翻译结果**，校对后再生成最终图片"
                        )
                        editor_headers = ["序号", "置信度", "坐标(xyxy)", "原文 (可修改)", "译文 (可修改)"]
                        editor_table = gr.Dataframe(
                            headers=editor_headers,
                            datatype=["number", "str", "str", "str", "str"],
                            label="翻译校对表格",
                            interactive=True,
                            row_count=(0, "dynamic"),
                            column_count=(5, "fixed"),
                            wrap=True,
                            max_height=380,
                        )

                    with gr.Tab("✅ 最终结果"):
                        with gr.Row():
                            image_cleaned = gr.Image(
                                label="步骤A：清除原文后",
                                type="pil",
                                height=340,
                            )
                            image_final = gr.Image(
                                label="步骤B：最终中文版",
                                type="pil",
                                height=340,
                            )
                        btn_download = gr.Button(
                            "💾 保存中文版到 output 文件夹",
                            variant="secondary",
                        )
                        file_output = gr.File(label="下载文件", height=60)

                status_box = gr.Textbox(
                    label="📋 运行状态",
                    lines=8,
                    elem_classes=["status-box"],
                )

        btn_ocr.click(
            fn=run_ocr_and_translate,
            inputs=[image_input, ocr_lang, translate_engine, source_lang],
            outputs=[image_preview, editor_table, status_box, session_id],
        )

        btn_generate.click(
            fn=apply_edits_and_generate,
            inputs=[image_input, session_id, editor_table, inpaint_method],
            outputs=[image_preview, image_cleaned, image_final, status_box],
        )

        btn_download.click(
            fn=save_output_image,
            inputs=[image_final, session_id],
            outputs=[file_output],
        )

        gr.Markdown(
            """
            ---
            ### 🚀 升级路线图
            - **V1.0**（当前）：OCR + 翻译 + 校对 + 擦除 + 嵌字 全流程打通
            - **V2.0**：批量处理 ZIP/PDF 多页漫画 + 气泡检测 + 字体自适应
            - **V3.0**：LaMa 高清擦除 + AI Inpainting + 多方向竖排文字识别
            """
        )

    return demo


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("[OK] Starting AI Comic Translator...")
    print(f"[INFO] Project dir : {os.path.dirname(os.path.abspath(__file__))}")
    print(f"[INFO] fonts/      : {config.FONT_DIR}")
    print(f"[INFO] input/      : {config.INPUT_DIR}")
    print(f"[INFO] output/     : {config.OUTPUT_DIR}")
    print(f"[INFO] temp/       : {config.TEMP_DIR}")
    print()

    demo = create_demo()
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="rose"),
        css="""
        .container { max-width: 1400px !important; }
        .status-box { background: #f8fafc; border-radius: 8px; padding: 12px; }
        """,
    )


if __name__ == "__main__":
    main()
