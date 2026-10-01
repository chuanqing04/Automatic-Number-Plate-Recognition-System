
import os
import sys
import argparse
import glob
import json
import logging
import time
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_pipeline.log"),
        logging.StreamHandler()
    ]
)

# ==============================================================================
# CLASS: PADDLE OCR ENGINE (From src.ocr.engine)
# ==============================================================================
try:
    from paddleocr import PaddleOCR
except ImportError:
    # If paddleocr is not installed, we can define a mock or error out. 
    # Current codebase assumes it is installed.
    PaddleOCR = None

class OCREngine:
    """Wrapper for PaddleOCR recognition-only mode."""

    def __init__(self, use_gpu: bool = False):
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is not installed. Please install it via pip install paddleocr.")

        # Initialize PaddleOCR in recognition-only mode
        if use_gpu:
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.set_device('gpu')
                    print("Info: Set Paddle device to GPU")
            except:
                pass

        # det=False disables detection (we assume cropped images)
        # rec=True enables recognition
        # cls=False disables angle classification (optional, but faster if plates are upright)
        # Disable document preprocessor when supported to avoid unwanted rotation/unwarp.
        ocr_kwargs = {
            "use_angle_cls": False,
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        try:
            self.ocr = PaddleOCR(**ocr_kwargs)
        except (TypeError, ValueError):
            ocr_kwargs.pop("use_doc_orientation_classify", None)
            ocr_kwargs.pop("use_doc_unwarping", None)
            self.ocr = PaddleOCR(**ocr_kwargs)

    def _run_ocr(self, image: np.ndarray, use_det: bool) -> list:
        """Run PaddleOCR with optional detection, fallback if args not supported."""
        try:
            return self.ocr.ocr(
                image,
                det=use_det,
                rec=True,
                cls=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except (TypeError, ValueError):
            return self.ocr.ocr(image)

    @staticmethod
    def _looks_like_box(obj: object) -> bool:
        if isinstance(obj, np.ndarray):
            if obj.shape == (4, 2):
                return True
            return False
        if not isinstance(obj, (list, tuple)) or len(obj) != 4:
            return False
        for pt in obj:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                return False
        return True

    @staticmethod
    def _normalize_box(obj: object) -> Optional[List[List[float]]]:
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            if obj.shape != (4, 2):
                return None
            return [[float(x), float(y)] for x, y in obj.tolist()]
        if not OCREngine._looks_like_box(obj):
            return None
        return [[float(pt[0]), float(pt[1])] for pt in obj]

    @staticmethod
    def _collect_entries(result: object) -> List[Dict]:
        entries: List[Dict] = []

        if isinstance(result, dict):
            rec_texts = result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', [])
            rec_polys = result.get('rec_polys') or result.get('rec_boxes')
            for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                box = None
                if isinstance(rec_polys, (list, tuple)) and idx < len(rec_polys):
                    box = OCREngine._normalize_box(rec_polys[idx])
                entries.append({"text": str(text), "score": float(score), "box": box})
            return entries

        if not isinstance(result, list):
            return entries

        for item in result:
            if isinstance(item, dict):
                rec_texts = item.get('rec_texts', [])
                rec_scores = item.get('rec_scores', [])
                rec_polys = item.get('rec_polys') or item.get('rec_boxes')
                for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    box = None
                    if isinstance(rec_polys, (list, tuple)) and idx < len(rec_polys):
                        box = OCREngine._normalize_box(rec_polys[idx])
                    entries.append({"text": str(text), "score": float(score), "box": box})
                continue

            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        text = sub_item.get("text", "")
                        score = sub_item.get("score", 0.0)
                        entries.append({"text": str(text), "score": float(score), "box": sub_item.get("box")})
                        continue

                    if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                        if OCREngine._looks_like_box(sub_item[0]) and isinstance(sub_item[1], (list, tuple)):
                            box = OCREngine._normalize_box(sub_item[0])
                            txt_obj = sub_item[1]
                            text = str(txt_obj[0])
                            score = float(txt_obj[1])
                            entries.append({"text": text, "score": score, "box": box})
                        else:
                            text = str(sub_item[0])
                            score = float(sub_item[1])
                            entries.append({"text": text, "score": score, "box": None})

        return entries

    @staticmethod
    def _score_text(text: str, conf: float) -> float:
        if not text:
            return -1.0
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        has_alpha = any(c.isalpha() for c in clean)
        has_digit = any(c.isdigit() for c in clean)
        score = conf
        if has_alpha and has_digit:
            score += 0.6
        score += min(len(clean), 8) * 0.05
        return score

    @staticmethod
    def _combine_multiline(entries: List[Dict]) -> Tuple[str, float]:
        with_box = [e for e in entries if e.get("box") is not None and e.get("text")]
        if len(with_box) < 2:
            return "", 0.0

        heights = []
        for e in with_box:
            ys = [pt[1] for pt in e["box"]]
            heights.append(max(ys) - min(ys))
        median_h = float(np.median(heights)) if heights else 0.0
        line_thresh = max(4.0, 0.6 * median_h)

        sortable = []
        for e in with_box:
            xs = [pt[0] for pt in e["box"]]
            ys = [pt[1] for pt in e["box"]]
            cx = float(sum(xs)) / 4.0
            cy = float(sum(ys)) / 4.0
            sortable.append((cy, cx, e))

        sortable.sort(key=lambda v: (v[0], v[1]))
        lines: List[List[Dict]] = []
        line_centers: List[float] = []

        for cy, cx, e in sortable:
            if not line_centers or abs(cy - line_centers[-1]) > line_thresh:
                lines.append([e])
                line_centers.append(cy)
            else:
                lines[-1].append(e)

        for line in lines:
            line.sort(key=lambda e: float(sum(pt[0] for pt in e["box"])) / 4.0)

        combined = "".join(e["text"] for line in lines for e in line)
        if not combined:
            return "", 0.0
        avg_score = float(np.mean([e["score"] for e in with_box])) if with_box else 0.0
        return combined, avg_score

    def recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on the image.
        Returns: (text, confidence)
        """
        result = self._run_ocr(image, use_det=True)
        entries = self._collect_entries(result)

        if not entries:
            result = self._run_ocr(image, use_det=False)
            entries = self._collect_entries(result)

        if not entries:
            return "", 0.0

        best_entry = max(entries, key=lambda e: e.get("score", 0.0))
        best_text = best_entry.get("text", "")
        best_score = float(best_entry.get("score", 0.0))

        combined_text, combined_score = self._combine_multiline(entries)

        if combined_text:
            best_val = self._score_text(best_text, best_score)
            combined_val = self._score_text(combined_text, combined_score)
            if combined_val > best_val:
                return combined_text, combined_score

        return best_text, best_score

    @staticmethod
    def fix_common_mistakes(text: str) -> str:
        """Fix specific OCR errors common in license plates."""
        replacements = {
            'O': '0', 'I': '1', 'B': '8', 'Q': '0',
            'D': '0', 'Z': '2', 'S': '5'
        }
        fixed = list(text)
        for i, char in enumerate(fixed):
            if char in replacements:
                prev_is_digit = i > 0 and fixed[i - 1].isdigit()
                next_is_digit = i + 1 < len(fixed) and fixed[i + 1].isdigit()
                if prev_is_digit and next_is_digit:
                    fixed[i] = replacements[char]
        return "".join(fixed)


# ==============================================================================
# CLASS: PREPROCESSOR & ENHANCER (From src.ocr.preprocess)
# ==============================================================================

class PlatePreprocessor:
    """Handles basic image preprocessing for noise reduction and morphological cleaning."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def remove_noise(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    @staticmethod
    def clean_morphology(image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closing


class PlateEnhancer:
    """Handles plate-specific enhancements like resizing and normalization."""

    def __init__(self, target_height: int = 64):
        self.target_height = target_height

    def resize_plate(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h == 0 or w == 0: return image
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        return cv2.resize(image, (new_width, self.target_height), interpolation=cv2.INTER_CUBIC)

    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def compute_readability(image: np.ndarray) -> float:
        return cv2.Laplacian(image, cv2.CV_64F).var()

    @staticmethod
    def pad_plate(image: np.ndarray, pad_value: int = 0, pad_ratio: float = 0.1) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = int(h * pad_ratio)
        pad_w = int(h * pad_ratio)
        return cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=pad_value)


# ==============================================================================
# CLASS: POST PROCESSOR (From src.ocr.postprocess)
# ==============================================================================

class PostProcessor:
    """Handles text cleanup and output formatting."""

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)

        # 1. Start with Alpha only rule? Check "A" suffix rule first.
        # User Rule: Suffix alphabets only allowed for 'W' series.
        if len(text) > 1 and text[0] != 'W':
             while text and text[-1].isalpha():
                 last = text[-1]
                 suffix_map = {'A': '4', 'B': '8', 'D': '0', 'G': '6', 'I': '1', 'L': '1', 'O': '0', 'Q': '0', 'S': '5', 'Z': '2'}
                 if last in suffix_map:
                     text = text[:-1] + suffix_map[last]
                     break 
                 else:
                     text = text[:-1]
        
        # 2. Max 4 Digits Logic
        match = re.match(r'^([A-Z]*)([0-9]+)([A-Z]*)$', text)
        if match:
            prefix, digits, suffix = match.groups()
            if len(digits) > 4:
                digit_to_alpha = {'8': 'B', '4': 'A', '0': 'D', '6': 'G', '5': 'S', '2': 'Z', '1': 'I', '7': 'Z', '3': 'J'}
                excess = len(digits) - 4
                digits_to_move = digits[:excess]
                digits_remain = digits[excess:]
                moved_alpha = ""
                for d in digits_to_move:
                    moved_alpha += digit_to_alpha.get(d, d) 
                text = prefix + moved_alpha + digits_remain + suffix

        # 3. Prevent alpha-digit-alpha-digit sequence in prefix groups
        digit_to_alpha_prefix = {
            '0': 'Q', '1': 'L', '2': 'Z', '3': 'J', '4': 'A',
            '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
        }
        match_alt = re.match(r'^([A-Z]+)([0-9]+)([A-Z]+)([0-9]+)$', text)
        if match_alt:
            prefix_letters, mid_digits, mid_letters, tail_digits = match_alt.groups()
            converted = "".join(digit_to_alpha_prefix.get(d, d) for d in mid_digits)
            text = f"{prefix_letters}{converted}{mid_letters}{tail_digits}"

        # 4. Prevent 4+ consecutive alphabets (convert or drop)
        alpha_to_digit = {
            'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'J': '1',
            'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'T': '7'
        }
        if text:
            cleaned = []
            alpha_run = 0
            for ch in text:
                if ch.isalpha():
                    alpha_run += 1
                    if alpha_run <= 3:
                        cleaned.append(ch)
                    else:
                        mapped = alpha_to_digit.get(ch)
                        if mapped:
                            cleaned.append(mapped)
                            alpha_run = 0
                        else:
                            alpha_run = 3
                else:
                    alpha_run = 0
                    cleaned.append(ch)
            text = "".join(cleaned)

        # 4a. Prevent alpha-digit-alpha in prefix (convert digit to letter)
        if len(text) >= 3 and re.match(r'^[A-Z][0-9][A-Z]', text):
            mapped = digit_to_alpha_prefix.get(text[1])
            if mapped:
                text = text[0] + mapped + text[2:]
        
        # 5. Sandwich Rules to fix mixed patterns
        def replace_middle_digit_context(match):
            a, digit, b = match.groups()
            digit_map = {'0': 'D', '8': 'B', '1': 'I', '5': 'S', '2': 'Z', '4': 'A', '6': 'G', '7': 'T', '3': 'J', 'Q': 'Q', 'O': 'D'}
            return f"{a}{digit_map.get(digit, digit)}{b}"

        def replace_middle_alpha_context(match):
            a, alpha, b = match.groups()
            alpha_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'J': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'T': '7'}
            return f"{a}{alpha_map.get(alpha, alpha)}{b}"

        for _ in range(3):
            old_text = text
            text = re.sub(r'([A-Z])([0-9])([A-Z])', replace_middle_digit_context, text)
            text = re.sub(r'([0-9])([A-Z])([0-9])', replace_middle_alpha_context, text)
            if text == old_text: break
        
        # 6. Enforce Start with Alpha (Malaysian rule)
        if len(text) > 0 and text[0].isdigit():
             first = text[0]
             digit_map = {'0': 'D', '8': 'B', '4': 'A', '5': 'S', '2': 'Z'}
             if first in digit_map:
                 text = digit_map[first] + text[1:]

        # 7. Fix 'AK0206' -> 'AKQ206'
        match_zero = re.match(r'^([A-Z]{2})0([0-9]+)$', text)
        if match_zero:
            prefix, rest = match_zero.groups()
            text = f"{prefix}Q{rest}"

        # 8. Final pass: plates do not use 'I'
        text = text.replace('I', '1')
            
        return text


# ==============================================================================
# CLASS: OCR PIPELINE (New Consolidated Logic)
# ==============================================================================

class OCRPipeline:
    """Encapsulates the multi-pass OCR strategy for license plate extraction."""

    def __init__(self, engine: Optional[OCREngine] = None):
        if engine:
            self.engine = engine
        else:
            try:
                self.engine = OCREngine(use_gpu=False)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OCR Engine: {e}")
        
        self.enhancer = PlateEnhancer()

    @staticmethod
    def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
        if gray.size == 0:
            return gray
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        mean_val = float(np.mean(blurred))
        thresh_type = cv2.THRESH_BINARY_INV if mean_val < 127 else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 31, 5
        )

    @staticmethod
    def _has_horizontal_gap(gray: np.ndarray) -> bool:
        if gray.size == 0 or gray.shape[0] < 10:
            return False
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(np.mean(th > 0))
        if white_ratio > 0.6:
            th = cv2.bitwise_not(th)
        row_sum = np.sum(th > 0, axis=1).astype(np.float32)
        if row_sum.max() <= 0:
            return False
        row_sum = cv2.GaussianBlur(row_sum.reshape(-1, 1), (1, 9), 0).ravel()
        mid_start = int(0.4 * len(row_sum))
        mid_end = int(0.6 * len(row_sum))
        if mid_end <= mid_start:
            return False
        valley = float(np.min(row_sum[mid_start:mid_end]))
        top_peak = float(np.max(row_sum[:mid_start])) if mid_start > 0 else 0.0
        bottom_peak = float(np.max(row_sum[mid_end:])) if mid_end < len(row_sum) else 0.0
        peak = float(np.max(row_sum))
        return valley < 0.3 * peak and top_peak > 0.55 * peak and bottom_peak > 0.55 * peak

    @staticmethod
    def _find_split_row(gray: np.ndarray) -> int:
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            return h // 2
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(np.mean(th > 0))
        if white_ratio > 0.6:
            th = cv2.bitwise_not(th)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        row_sum = np.sum(th > 0, axis=1).astype(np.float32)
        if row_sum.max() <= 0:
            return h // 2
        row_sum = cv2.GaussianBlur(row_sum.reshape(-1, 1), (1, 9), 0).ravel()
        min_row = max(10, int(0.2 * h))
        max_row = min(h - 10, int(0.8 * h))
        if max_row <= min_row:
            return h // 2
        window = row_sum[min_row:max_row]
        split = min_row + int(np.argmin(window))
        valley = float(row_sum[split])
        peak = float(np.max(row_sum))
        if peak > 0 and valley > 0.5 * peak:
            return h // 2
        return split

    def _should_try_split(self, gray: np.ndarray) -> bool:
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return False
        aspect = w / h
        return aspect < 2.8 or self._has_horizontal_gap(gray)

    def process_image(self, image: np.ndarray, filename: str = "image") -> Dict:
        """Runs the full OCR pipeline on a given image."""
        
        # 1. Preprocessing (Grayscale)
        gray = PlatePreprocessor.to_grayscale(image)
        
        # 2. Pipeline Execution (Multi-strategy)
        # Strategy: Resize -> Try Standard, Inverted, Padded, HistEq, Split
        
        # 0. Resize
        resized_gray = self.enhancer.resize_plate(gray)
        resized_input = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR) # Back to BGR for Paddle
        
        # Pass 1: Resized Normal
        text_orig, conf_orig = self.engine.recognize_text(resized_input)
        
        # Pass 2: Resized Inverted
        inverted = cv2.bitwise_not(resized_input)
        text_inv, conf_inv = self.engine.recognize_text(inverted)

        early_candidates = [(text_orig, conf_orig), (text_inv, conf_inv)]

        # Condition: "cannot detect text for two passes" → both empty
        no_text_two_passes = all(t.strip() == "" for t, _ in early_candidates)

        # Condition: "confidence < 80%" → check best from early candidates
        best_conf = max(c for _, c in early_candidates)
        low_conf = best_conf < 0.80

        # Attempt a try split here to determine whether to end early or continue
        h, w = image.shape[:2]
        try_split = h > 0 and self._should_try_split(gray)

        if (no_text_two_passes or low_conf) and not try_split:
            return {
                "file": filename,
                "status": "skipped",
                "reason": "weak_early_ocr",
                "plate_text": "",
                "confidence": float(best_conf),
            }
        
        # Pass 3: Resized Padded (Black border)
        padded = self.enhancer.pad_plate(resized_input, pad_value=0, pad_ratio=0.2)
        text_pad, conf_pad = self.engine.recognize_text(padded)
        
        # Pass 4: High-Res (Height 128)
        if h > 0:
            aspect_ratio = w / h
            new_width_large = int(128 * aspect_ratio)
            resized_large_gray = cv2.resize(gray, (new_width_large, 128), interpolation=cv2.INTER_CUBIC)
            resized_large_bgr = cv2.cvtColor(resized_large_gray, cv2.COLOR_GRAY2BGR)
            text_large, conf_large = self.engine.recognize_text(resized_large_bgr)
            hist_eq_gray = cv2.equalizeHist(resized_large_gray)
            hist_eq = cv2.cvtColor(hist_eq_gray, cv2.COLOR_GRAY2BGR)
            text_hist, conf_hist = self.engine.recognize_text(hist_eq)
        else:
            text_large, conf_large = "", 0.0
            text_hist, conf_hist = "", 0.0

        # Pass 5: Adaptive Threshold (Height 96)
        if h > 0:
            aspect_ratio = w / h
            new_width_mid = int(96 * aspect_ratio)
            resized_mid_gray = cv2.resize(gray, (new_width_mid, 96), interpolation=cv2.INTER_CUBIC)
            adaptive = self._adaptive_threshold(resized_mid_gray)
            adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
            text_adapt, conf_adapt = self.engine.recognize_text(adaptive_bgr)
        else:
            text_adapt, conf_adapt = "", 0.0

        # Pass 6: Square Plate Split (Heuristic for Multi-row plates)
        text_split = ""
        conf_split = 0.0
        
        if try_split:
            logging.info("Multi-row plate hint detected. Attempting Split Strategy.")
            text_split, conf_split = self._process_split_plate(gray)

        # Selection Logic
        candidates = [
            ("Resized_Original", text_orig, conf_orig),
            ("Resized_Inverted", text_inv, conf_inv),
            ("Resized_Padded", text_pad, conf_pad),
            ("Resized_Large", text_large, conf_large),
            ("Resized_HistEq", text_hist, conf_hist),
            ("Adaptive_Thresh", text_adapt, conf_adapt),
            ("Square_Split", text_split, conf_split)
        ]

        best_variant, best_text, best_conf, best_score = self._select_best_result(candidates)
        logging.info(f"Selected ({best_variant}): '{best_text}' (Conf: {best_conf:.2f})")

        # Fix Mistakes
        fixed_text = OCREngine.fix_common_mistakes(best_text)

        # Post-Processing
        final_text = PostProcessor.clean_text(fixed_text)

        return {
            "file": filename,
            "status": "success",
            "plate_text": final_text,
            "confidence": float(best_conf),
            "method": best_variant,
            "readability": self.enhancer.compute_readability(resized_gray)
        }

    def _process_split_plate(self, gray_image: np.ndarray) -> Tuple[str, float]:
        h, w = gray_image.shape[:2]
        split_row = self._find_split_row(gray_image)
        min_band = max(10, int(0.2 * h))
        if split_row < min_band or (h - split_row) < min_band:
            split_row = h // 2
        top_half = gray_image[0:split_row, :]
        bottom_half = gray_image[split_row:h, :]
        
        def resize_to_height(img, height):
            h, w = img.shape[:2]
            if h == 0 or w == 0: return img
            ar = w / h
            new_w = int(height * ar)
            return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_CUBIC)

        top_half_resized = resize_to_height(top_half, 96)
        top_half_large = resize_to_height(top_half, 192)
        bottom_half_resized = resize_to_height(bottom_half, 96)
        
        bot_bgr_input = cv2.cvtColor(bottom_half_resized, cv2.COLOR_GRAY2BGR)
        bot_padded = self.enhancer.pad_plate(bot_bgr_input, pad_value=255, pad_ratio=0.15)
        bot_bgr = bot_padded

        top_bgr = cv2.cvtColor(top_half_resized, cv2.COLOR_GRAY2BGR)
        top_large_bgr = cv2.cvtColor(top_half_large, cv2.COLOR_GRAY2BGR)
        
        t1, c1 = self.engine.recognize_text(top_bgr) # Variant 1
        t1b, c1b = self.engine.recognize_text(top_large_bgr) # Variant 1b (hi-res)
        
        _, thresh = cv2.threshold(top_half_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        t2, c2 = self.engine.recognize_text(thresh_bgr) # Variant 2
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(top_half_resized)
        clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
        t3, c3 = self.engine.recognize_text(clahe_bgr) # Variant 3
        
        inverted_top = cv2.bitwise_not(top_bgr)
        t4, c4 = self.engine.recognize_text(inverted_top) # Variant 4

        adaptive_top = self._adaptive_threshold(top_half_resized)
        adaptive_top_bgr = cv2.cvtColor(adaptive_top, cv2.COLOR_GRAY2BGR)
        t5, c5 = self.engine.recognize_text(adaptive_top_bgr) # Variant 5

        _, thresh_large = cv2.threshold(top_half_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_large_bgr = cv2.cvtColor(thresh_large, cv2.COLOR_GRAY2BGR)
        t6, c6 = self.engine.recognize_text(thresh_large_bgr) # Variant 6 (hi-res Otsu)
        inverted_large = cv2.bitwise_not(top_large_bgr)
        t7, c7 = self.engine.recognize_text(inverted_large) # Variant 7 (hi-res inverted)
        
        top_candidates = [
            (t1, c1),
            (t1b, c1b),
            (t2, c2),
            (t3, c3),
            (t4, c4),
            (t5, c5),
            (t6, c6),
            (t7, c7),
        ]
        best_top_text = ""
        best_top_score_val = -1.0
        best_top_conf = 0.0
        
        for t, conf in top_candidates:
            if not t: continue
            s = self.engine._score_text(t, conf)
            if t.isalpha(): s += 0.4
            if any(c.isdigit() for c in t): s -= 0.1
            if s > best_top_score_val:
                best_top_score_val = s
                best_top_text = t
                best_top_conf = conf
                
        text_top = best_top_text
        conf_top = best_top_conf if best_top_text else 0.0

        bot_candidates = []
        b1, cb1 = self.engine.recognize_text(bot_bgr)
        bot_candidates.append((b1, cb1))
        inverted_bot = cv2.bitwise_not(bot_bgr)
        b2, cb2 = self.engine.recognize_text(inverted_bot)
        bot_candidates.append((b2, cb2))
        adaptive_bot = self._adaptive_threshold(bottom_half_resized)
        adaptive_bot_bgr = cv2.cvtColor(adaptive_bot, cv2.COLOR_GRAY2BGR)
        b3, cb3 = self.engine.recognize_text(adaptive_bot_bgr)
        bot_candidates.append((b3, cb3))
        _, bot_otsu = cv2.threshold(bottom_half_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bot_otsu_bgr = cv2.cvtColor(bot_otsu, cv2.COLOR_GRAY2BGR)
        b4, cb4 = self.engine.recognize_text(bot_otsu_bgr)
        bot_candidates.append((b4, cb4))
        _, bot_otsu_inv = cv2.threshold(bottom_half_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bot_otsu_inv_bgr = cv2.cvtColor(bot_otsu_inv, cv2.COLOR_GRAY2BGR)
        b5, cb5 = self.engine.recognize_text(bot_otsu_inv_bgr)
        bot_candidates.append((b5, cb5))

        best_bot_text = ""
        best_bot_conf = 0.0
        best_bot_score_val = -1.0
        for t, conf in bot_candidates:
            if not t: continue
            s = self.engine._score_text(t, conf)
            if t.isdigit(): s += 0.4
            if any(c.isalpha() for c in t): s -= 0.1
            if len(t) >= 4: s += 0.2
            if s > best_bot_score_val:
                best_bot_score_val = s
                best_bot_text = t
                best_bot_conf = conf

        text_bot = best_bot_text
        conf_bot = best_bot_conf
        text_split = f"{text_top}{text_bot}"
        conf_split = (conf_top + conf_bot) / 2 if (text_top or text_bot) else 0.0
        return text_split, conf_split

    def _select_best_result(self, candidates: List[Tuple[str, str, float]]) -> Tuple[str, str, float, float]:
        best_variant = candidates[0][0]
        best_text = candidates[0][1]
        best_conf = candidates[0][2]
        best_score = -1.0
        for variant, text, conf in candidates:
            score = self._score_variant(text, conf)
            if score > best_score:
                best_score = score
                best_text = text
                best_conf = conf
                best_variant = variant
        return best_variant, best_text, best_conf, best_score

    def _score_variant(self, text: str, conf: float) -> float:
        if not text: return 0.0
        has_alpha = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        if not (has_alpha and has_digit): return 0.0
        score = conf * 100
        if has_alpha and has_digit: score += 50 
        score += len(text) * 5
        return score


# ==============================================================================
# ENTRY POINT AND CLI
# ==============================================================================

# Global Pipeline Instance
_PIPELINE_INSTANCE = None

def get_pipeline():
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        try:
            import paddle
            use_gpu = paddle.device.is_compiled_with_cuda()
            if use_gpu: 
                 paddle.device.set_device('gpu')
        except:
            use_gpu = False
            
        engine = OCREngine(use_gpu=use_gpu)
        _PIPELINE_INSTANCE = OCRPipeline(engine=engine)
    return _PIPELINE_INSTANCE

def extract_text(processed_image_path: str, engine: OCREngine = None) -> dict:
    """
    Takes a processed image path, runs the robust multi-pass OCR logic, 
    and returns a dict with 'plate_text' and other metadata.
    Compatibility wrapper for GUI.
    """
    if not os.path.exists(processed_image_path):
        raise FileNotFoundError(f"Processed image not found: {processed_image_path}")

    image = cv2.imread(processed_image_path)
    if image is None:
        raise ValueError(f"Failed to load processed image: {processed_image_path}")

    if engine:
        pipeline = OCRPipeline(engine=engine)
    else:
        pipeline = get_pipeline()

    return pipeline.process_image(image, filename=os.path.basename(processed_image_path))


def process_single_image_cli(image_path: str, result_dir: str, pipeline: OCRPipeline) -> dict:
    filename = os.path.basename(image_path)
    file_stem = Path(image_path).stem
    logging.info(f"Processing: {filename}")
    
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return {"file": filename, "status": "failed", "reason": "load_error"}

    result = pipeline.process_image(image, filename)
    
    json_path = os.path.join(result_dir, f"{file_stem}.json")
    with open(json_path, 'w') as f:
        json.dump({"plate_text": result["plate_text"]}, f, indent=2)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR License Plate Extraction Pipeline (Standalone)")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    args = parser.parse_args()

    # Determine Input Directory
    if os.path.exists(os.path.join("..", "data", "raw")):
        data_base = os.path.join("..", "data")
    else:
        data_base = "data" 
        
    result_base = "." 
    raw_dir = os.path.join(data_base, "raw")
    output_dir = os.path.join(result_base, "results/ocr_output")
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = args.input if args.input != "data/raw" else raw_dir
    
    logging.info("Initializing PaddleOCR Engine...")
    pipeline = get_pipeline()

    # Gather images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        logging.warning(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting processing...")
    results = []
    start_time = time.time()
    
    for img_path in tqdm(image_files, desc="Processing Plates"):
        res = process_single_image_cli(img_path, output_dir, pipeline)
        results.append(res)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(results) if results else 0
    
    report_path = os.path.join(result_base, "ocr_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Total Images   : {len(results)}")
    print(f"Total Time     : {total_time:.2f} seconds")
    print(f"Avg Time/Image : {avg_time:.4f} seconds")

if __name__ == "__main__":
    main()
