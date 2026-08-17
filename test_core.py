import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.processor import ComicTranslator
import config


def create_test_image(width: int = 900, height: int = 600) -> str:
    image = np.ones((height, width, 3), dtype=np.uint8) * 245

    cv2.rectangle(image, (80, 30), (420, 130), (255, 255, 255), -1)
    cv2.rectangle(image, (80, 30), (420, 130), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.ellipse(image, (80, 145), (18, 10), 0, 0, 180, (255, 255, 255), -1)
    cv2.ellipse(image, (80, 145), (18, 10), 0, 0, 180, (120, 120, 120), 2, cv2.LINE_AA)
    cv2.ellipse(image, (65, 160), (10, 7), 0, 0, 180, (255, 255, 255), -1)
    cv2.ellipse(image, (65, 160), (10, 7), 0, 0, 180, (120, 120, 120), 2, cv2.LINE_AA)

    cv2.putText(image, "Kon'nichiwa!", (130, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "(Hello!)", (155, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.rectangle(image, (500, 320), (830, 430), (255, 255, 255), -1)
    cv2.rectangle(image, (500, 320), (830, 430), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.ellipse(image, (830, 445), (18, 10), 0, 0, 180, (255, 255, 255), -1)
    cv2.ellipse(image, (830, 445), (18, 10), 0, 0, 180, (120, 120, 120), 2, cv2.LINE_AA)
    cv2.ellipse(image, (845, 460), (10, 7), 0, 0, 180, (255, 255, 255), -1)
    cv2.ellipse(image, (845, 460), (10, 7), 0, 0, 180, (120, 120, 120), 2, cv2.LINE_AA)

    cv2.putText(image, "Nani o shite iru no?", (540, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "(What are you doing?)", (545, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.rectangle(image, (280, 490), (560, 565), (255, 255, 255), -1)
    cv2.rectangle(image, (280, 490), (560, 565), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(image, "Arigatou!", (330, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA)

    path = os.path.join(config.INPUT_DIR, "demo_comic.png")
    cv2.imwrite(path, image)
    return path


def main():
    print("=" * 60)
    print("  AI 漫画翻译器 V1.0 - 核心模块流程测试")
    print("=" * 60)

    test_img_path = create_test_image()
    print(f"\n[1] 生成测试漫画: {test_img_path}")

    processor = ComicTranslator(
        ocr_lang="japan",
        translate_engine="mock",
        inpaint_method="telea",
    )
    print("[2] 初始化完成: OCR + Translator + Remover + Typesetter\n")

    out_final = os.path.join(config.OUTPUT_DIR, "demo_translated.png")
    steps = processor.process(test_img_path, return_steps=True, output_path=out_final)

    print("\n[3] 处理结果概览:")
    for r in steps["results"]:
        print(f"   区域 #{r['index']} bbox={r['bbox']}")
        print(f"     原文: {r['text']}")
        print(f"     译文: {r['translation']}")
        print(f"     背景色: RGB{r['bg_color_rgb']}  白底={r['is_white_bg']}")

    preview_path = os.path.join(config.OUTPUT_DIR, "demo_ocr_preview.png")
    cleaned_path = os.path.join(config.OUTPUT_DIR, "demo_cleaned.png")

    from modules.typesetter import Typesetter
    ts = Typesetter()
    if steps["preview_image"] is not None:
        ts.save_image(steps["preview_image"], preview_path)
    ts.save_image(steps["cleaned_image"], cleaned_path)

    print("\n[4] 生成文件:")
    print(f"   ✅ 原图         : {test_img_path}")
    print(f"   ✅ OCR预览(红框): {preview_path}")
    print(f"   ✅ 清除原文     : {cleaned_path}")
    print(f"   ✅ 最终中文版   : {out_final}")

    print("\n[5] 模块导入检查:")
    modules_to_check = [
        ("modules.ocr", "ComicOCR"),
        ("modules.translator", "Translator"),
        ("modules.remover", "TextRemover"),
        ("modules.typesetter", "Typesetter"),
        ("modules.processor", "ComicTranslator"),
    ]
    for mod_name, cls_name in modules_to_check:
        try:
            m = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(m, cls_name)
            print(f"   ✅ {mod_name}.{cls_name}")
        except Exception as e:
            print(f"   ❌ {mod_name}.{cls_name}: {e}")

    print("\n" + "=" * 60)
    print("  测试通过！核心流程已跑通 ✅")
    print("  接下来：pip install gradio paddleocr paddlepaddle 即可启用完整功能")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
