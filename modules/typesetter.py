import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Typesetter:
    def __init__(
        self,
        font_path: str = None,
        font_size: int = None,
        text_color: Tuple[int, int, int] = None,
    ):
        self.font_path = font_path or self._find_font()
        self.default_font_size = font_size or config.DEFAULT_FONT_SIZE
        self.default_text_color = text_color or config.DEFAULT_TEXT_COLOR
        self._font_cache = {}

    def _find_font(self) -> str:
        candidates = [
            config.DEFAULT_FONT_REGULAR,
            config.DEFAULT_FONT_BOLD,
            os.path.join(config.FONT_DIR, "NotoSansSC-Regular.otf"),
            os.path.join(config.FONT_DIR, "msyh.ttc"),
            os.path.join(config.FONT_DIR, "msyh.ttf"),
            os.path.join(config.FONT_DIR, "simhei.ttf"),
            os.path.join(config.FONT_DIR, "simsun.ttc"),
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def _get_font(self, font_size: int, font_path: str = None) -> Optional[ImageFont.FreeTypeFont]:
        path = font_path or self.font_path
        key = f"{path}_{font_size}"

        if key in self._font_cache:
            return self._font_cache[key]

        try:
            if path and os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
            else:
                font = ImageFont.load_default()
            self._font_cache[key] = font
            return font
        except Exception as e:
            print(f"[排版 警告] 加载字体失败 size={font_size}: {e}")
            try:
                return ImageFont.load_default()
            except Exception:
                return None

    def _measure_text(
        self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
    ) -> Tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            try:
                w, h = draw.textsize(text, font=font)
                return (w, h)
            except Exception:
                return (len(text) * (font.size // 2), font.size)

    def _split_lines(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        font: ImageFont.FreeTypeFont,
    ) -> List[str]:
        if not text:
            return []

        lines = []
        current = ""
        font_size = font.size

        for char in text:
            if char in ("\n", "\r"):
                if current:
                    lines.append(current)
                    current = ""
                continue

            test = current + char
            w, _ = self._measure_text(draw, test, font)
            if w <= max_width or not current:
                current = test
            else:
                lines.append(current)
                current = char

        if current:
            lines.append(current)

        if not lines:
            lines = [text]
        return lines

    def _auto_fit_font_size(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box_width: int,
        box_height: int,
        font_path: str = None,
    ) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
        min_size = config.MIN_FONT_SIZE
        max_size = config.MAX_FONT_SIZE
        best_font = None
        best_lines = []
        best_size = min_size

        for size in range(max_size, min_size - 1, -2):
            font = self._get_font(size, font_path)
            if font is None:
                continue

            lines = self._split_lines(draw, text, box_width - 8, font)
            if not lines:
                continue

            line_height = size + 2
            total_h = len(lines) * line_height
            max_line_w = 0
            for line in lines:
                w, _ = self._measure_text(draw, line, font)
                max_line_w = max(max_line_w, w)

            if total_h <= box_height - 8 and max_line_w <= box_width - 8:
                best_font = font
                best_lines = lines
                best_size = size
                break

        if best_font is None:
            best_font = self._get_font(min_size, font_path)
            best_lines = self._split_lines(draw, text, box_width - 8, best_font)
            best_size = min_size

        return best_font, best_lines, best_size

    def _cv2_to_pil(self, image: np.ndarray) -> Image.Image:
        if len(image.shape) == 2:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def add_text(
        self,
        image: Union[str, np.ndarray, Image.Image],
        items: List[Dict[str, Any]],
        padding: int = 6,
    ) -> np.ndarray:
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = self._cv2_to_pil(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError("image 必须是文件路径、numpy.ndarray 或 PIL.Image")

        draw = ImageDraw.Draw(pil_img)

        for item in items:
            bbox = item.get("bbox")
            text = item.get("translation") or item.get("text") or ""
            if not bbox or len(bbox) < 4 or not text:
                continue

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if w <= 10 or h <= 10:
                continue

            text_color = item.get("text_color_rgb") or item.get("text_color") or self.default_text_color
            font_path = item.get("font_path") or self.font_path
            bg_color = item.get("bg_color_rgb")

            box_w = max(20, w - padding * 2)
            box_h = max(16, h - padding * 2)

            font, lines, font_size = self._auto_fit_font_size(draw, text, box_w, box_h, font_path)
            if font is None:
                continue

            line_spacing = max(2, font_size // 6)

            # 先按真实行高测量，再整体垂直居中，避免视觉偏移
            measured = []
            for line in lines:
                lw, lh = self._measure_text(draw, line, font)
                measured.append((line, lw, lh))
            total_h = sum(lh for _, _, lh in measured) + line_spacing * (len(measured) - 1)
            start_y = y1 + padding + max(0, (box_h - total_h) / 2)

            # 描边：深色文字加白边、浅色文字加黑边，提升在复杂背景上的可读性
            try:
                lum = (
                    0.299 * text_color[0]
                    + 0.587 * text_color[1]
                    + 0.114 * text_color[2]
                )
            except Exception:
                lum = 0
            stroke_color = (255, 255, 255) if lum < 160 else (0, 0, 0)
            stroke_w = 2 if font_size >= 16 else 1

            cursor_y = start_y
            for line, lw, lh in measured:
                offset_x = x1 + padding + max(0, (box_w - lw) / 2)
                offset_y = cursor_y

                try:
                    if bg_color and len(bg_color) == 3:
                        pad = 2
                        rect_x1 = int(offset_x - pad)
                        rect_y1 = int(offset_y - pad)
                        rect_x2 = int(offset_x + lw + pad)
                        rect_y2 = int(offset_y + lh + pad)
                        draw.rectangle(
                            [rect_x1, rect_y1, rect_x2, rect_y2],
                            fill=tuple(bg_color),
                        )

                    draw.text(
                        (offset_x, offset_y),
                        line,
                        font=font,
                        fill=tuple(text_color),
                        stroke_width=stroke_w,
                        stroke_fill=stroke_color,
                    )
                except Exception as e:
                    print(f"[排版 警告] 绘制文本失败: {e}")
                cursor_y += lh + line_spacing

        return self._pil_to_cv2(pil_img)

    def draw_bboxes(
        self,
        image: Union[str, np.ndarray],
        items: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        show_text: bool = True,
    ) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"无法加载图片: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError("image 必须是文件路径或 numpy.ndarray")

        h, w = img.shape[:2]
        labels = []
        for i, item in enumerate(items, 1):
            bbox = item.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            if show_text:
                text = item.get("text") or item.get("translation") or f"#{i}"
                labels.append((x1, y1, f"{i}: {text[:12]}"))

        # Labels may contain Japanese/Chinese, which cv2.putText cannot render
        # reliably (Hershey fonts are ASCII-only). Draw them with PIL + CJK font.
        if labels:
            pil_img = self._cv2_to_pil(img)
            draw = ImageDraw.Draw(pil_img)
            font_size = max(14, int(h * 0.016))
            font = self._get_font(font_size, self.font_path)
            rgb = tuple(color[::-1])
            for x1, y1, label in labels:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                bx, by = x1, max(th + 4, y1 - 4)
                draw.rectangle(
                    [bx, by - th - 6, bx + tw + 8, by + 2],
                    fill=(255, 255, 255),
                )
                draw.text((bx + 4, by - th - 4), label, font=font, fill=rgb)
            img = self._pil_to_cv2(pil_img)

        return img

    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            return cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"[排版 错误] 保存图片失败: {e}")
            return False
