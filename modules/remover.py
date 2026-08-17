import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TextRemover:
    def __init__(self, method: str = None, radius: int = None):
        self.method = (method or config.INPAINT_METHOD).lower()
        self.radius = radius or config.INPAINT_RADIUS

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"无法加载图片: {image}")
            return img
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image.copy()
        else:
            raise TypeError("image 必须是文件路径或 numpy.ndarray")

    def _build_mask(
        self,
        image_shape: Tuple[int, int],
        boxes: List[Tuple[int, int, int, int]],
        dilate: int = 2,
    ) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        if dilate and dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate * 2 + 1, dilate * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _detect_bubble_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return (255, 255, 255)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return (255, 255, 255)

        pixels = np.float32(roi.reshape(-1, 3))
        if len(pixels) > 5000:
            idx = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[idx]

        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 3
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            unique, counts = np.unique(labels, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]
            color = centers[dominant_idx]
            return (int(color[0]), int(color[1]), int(color[2]))
        except Exception:
            avg = np.mean(pixels, axis=0)
            return (int(avg[0]), int(avg[1]), int(avg[2]))

    def remove_text(
        self,
        image: Union[str, np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        return_colors: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]]:
        img = self._load_image(image)
        if not boxes:
            if return_colors:
                return img, []
            return img

        mask = self._build_mask(img.shape, boxes, dilate=2)

        if self.method == "telea":
            inpaint_flag = cv2.INPAINT_TELEA
        elif self.method == "ns":
            inpaint_flag = cv2.INPAINT_NS
        else:
            inpaint_flag = cv2.INPAINT_TELEA

        result = cv2.inpaint(img, mask, max(1, self.radius), inpaint_flag)

        if return_colors:
            bubble_info = []
            for box in boxes:
                color_bgr = self._detect_bubble_color(img, box)
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                is_white = (
                    color_bgr[0] > 220 and color_bgr[1] > 220 and color_bgr[2] > 220
                )
                bubble_info.append(
                    {
                        "bbox": box,
                        "bg_color_bgr": color_bgr,
                        "bg_color_rgb": color_rgb,
                        "is_white_bg": is_white,
                        "text_color_rgb": (0, 0, 0) if is_white else (0, 0, 0),
                    }
                )
            return result, bubble_info

        return result

    def remove_text_simple(
        self,
        image: Union[str, np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        img = self._load_image(image)
        if not boxes:
            return img

        h, w = img.shape[:2]
        result = img.copy()

        for box in boxes:
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            color = self._detect_bubble_color(img, box)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

        return result

    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            return cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"[清除 错误] 保存图片失败: {e}")
            return False
