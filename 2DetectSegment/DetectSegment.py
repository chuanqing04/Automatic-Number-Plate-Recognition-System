from __future__ import annotations

from typing import Optional, Tuple, List

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points).reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _score_quad(quad: np.ndarray, area: float, roi_w: int, roi_h: int) -> float:
    cx = float(np.mean(quad[:, 0])) / max(1.0, roi_w)
    center_bonus = 1.0 - abs(cx - 0.5) * 2.0  # 1 at center, 0 at edges
    center_bonus = max(0.0, center_bonus)

    roi_area = float(roi_w * roi_h)
    return (area / roi_area) + 0.25 * center_bonus


def _collect_quads_from_contours(
    closed: np.ndarray, roi_w: int, roi_h: int, limit: int = 12
) -> List[Tuple[float, np.ndarray]]:
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
    roi_area = float(roi_w * roi_h)

    candidates: List[Tuple[float, np.ndarray]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 0.0008 * roi_area or area > 0.75 * roi_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        quad = order_points(approx)

        w1 = np.linalg.norm(quad[1] - quad[0])
        w2 = np.linalg.norm(quad[2] - quad[3])
        h1 = np.linalg.norm(quad[3] - quad[0])
        h2 = np.linalg.norm(quad[2] - quad[1])

        width = max(w1, w2)
        height = max(h1, h2)
        if height <= 1:
            continue

        aspect = width / height
        if aspect < 0.85 or aspect > 12.0:
            continue

        if width < 0.12 * roi_w or height < 0.04 * roi_h:
            continue

        score = _score_quad(quad, area, roi_w, roi_h)
        candidates.append((score, quad))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:limit]


def _collect_quads_from_sobel(
    gray: np.ndarray, roi_w: int, roi_h: int, limit: int = 10
) -> List[Tuple[float, np.ndarray]]:
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradx = np.absolute(gradx)
    gradx = (255 * (gradx / (gradx.max() + 1e-6))).astype(np.uint8)
    gradx = cv2.GaussianBlur(gradx, (5, 5), 0)

    _, th = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blob = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    blob = cv2.erode(blob, None, iterations=1)
    blob = cv2.dilate(blob, None, iterations=2)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]
    roi_area = float(roi_w * roi_h)

    candidates: List[Tuple[float, np.ndarray]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 0.0008 * roi_area or area > 0.80 * roi_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect < 0.85 or aspect > 12.0:
            continue

        if max(rw, rh) < 0.12 * roi_w or min(rw, rh) < 0.04 * roi_h:
            continue

        box = cv2.boxPoints(rect)
        quad = order_points(box)

        score = _score_quad(quad, area, roi_w, roi_h)
        candidates.append((score, quad))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:limit]


def _validate_plate_region(image: np.ndarray, quad: np.ndarray) -> Tuple[bool, float]:
    """
    Heuristic validation that the quad likely contains a plate.
    Returns (is_valid, quality_score).
    """
    patch = crop_plate(image, quad)
    if patch is None or patch.size == 0:
        return False, -1e9

    h, w = patch.shape[:2]
    if h < 15 or w < 40:
        return False, -1e9

    # Do not upscale the image for validation to avoid zooming artifacts
    eval_patch = patch
    if w > 360:
        scale = 360.0 / float(w)
        eval_patch = cv2.resize(
            patch, (360, max(4, int(h * scale))), interpolation=cv2.INTER_AREA
        )

    if len(eval_patch.shape) == 2:
        gray = eval_patch
    else:
        gray = cv2.cvtColor(eval_patch, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray)) / 255.0

    edges = cv2.Canny(gray, 60, 180)
    edge_ratio = float(np.count_nonzero(edges)) / max(1.0, edges.size)

    col_sum = np.sum(edges > 0, axis=0).astype(np.float32)
    row_sum = np.sum(edges > 0, axis=1).astype(np.float32)
    col_coverage = float(np.count_nonzero(col_sum > 0)) / max(1.0, edges.shape[1])

    col_var = float(np.std(col_sum)) / (float(np.mean(col_sum)) + 1e-6)
    row_var = float(np.std(row_sum)) / (float(np.mean(row_sum)) + 1e-6)

    is_valid = (
        0.01 <= edge_ratio <= 0.45
        and contrast >= 0.05
        and 0.15 <= col_coverage <= 0.98
        and col_var >= 0.08
        and row_var >= 0.06
    )

    quality = (
        edge_ratio * 0.6
        + contrast * 0.8
        + col_var * 0.3
        + row_var * 0.2
        + max(0.0, (col_coverage - 0.25)) * 0.1
    )
    return is_valid, quality


def _quad_to_bbox(quad: np.ndarray) -> Tuple[float, float, float, float]:
    xs = quad[:, 0]
    ys = quad[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _compute_iou(
    box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]
) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _generate_rois(W: int, H: int) -> List[Tuple[int, int, int, int]]:
    """
    Create a balanced ROI set: 10 rectangular bands + 10 square windows.
    """
    rois: List[Tuple[int, int, int, int]] = []

    def add_roi(x0: int, y0: int, x1: int, y1: int) -> None:
        roi = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            return
        if roi not in rois:
            rois.append(roi)

    # 10 rectangular bands (full width, varying heights/offsets)
    rect_heights = [0.30, 0.40, 0.50, 0.60]
    rect_offsets = [0.0, 0.2, 0.4]
    for hf in rect_heights:
        roi_h = max(1, int(H * hf))
        max_y = max(1, H - roi_h)
        for oy in rect_offsets:
            y0 = int(oy * max_y)
            add_roi(0, y0, W, y0 + roi_h)
            if len(rois) >= 10:
                break
        if len(rois) >= 10:
            break

    # 10 square windows (varying size/position)
    square_fracs = [0.60, 0.50, 0.40, 0.35]
    square_offsets = [0.0, 0.25, 0.5]
    for frac in square_fracs:
        side = max(1, int(min(W, H) * frac))
        max_x = max(1, W - side)
        max_y = max(1, H - side)
        for ox in square_offsets:
            for oy in square_offsets:
                add_roi(int(ox * max_x), int(oy * max_y), int(ox * max_x) + side, int(oy * max_y) + side)
                if len(rois) >= 20:
                    return rois

    return rois[:20]


def detect_plate_candidates(
    image: np.ndarray, max_candidates: int = 5, iou_threshold: float = 0.25
) -> List[np.ndarray]:
    """
    Return multiple plate quad candidates ordered by confidence, skipping overlaps
    so retries check new regions instead of the same area.
    """
    if image is None or image.size == 0:
        return []

    H, W = image.shape[:2]

    rois: list[Tuple[int, int, int, int]] = _generate_rois(W, H)

    raw_candidates: List[Tuple[float, np.ndarray]] = []

    for (x0, y0, x1, y1) in rois:
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]

        if len(roi.shape) == 2:
            gray = roi
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        edges = cv2.Canny(gray, 40, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        for score, quad in _collect_quads_from_contours(closed, roi_w, roi_h):
            raw_candidates.append((score, quad + np.array([x0, y0], dtype=np.float32)))

        for score, quad in _collect_quads_from_sobel(gray, roi_w, roi_h):
            raw_candidates.append((score, quad + np.array([x0, y0], dtype=np.float32)))

    if not raw_candidates:
        return []

    scored: List[Tuple[float, np.ndarray]] = []
    for base_score, quad in raw_candidates:
        quad = order_points(quad)
        is_valid, quality = _validate_plate_region(image, quad)
        combined = base_score + (quality if is_valid else quality * 0.25)
        scored.append((combined, quad))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[float, float, float, float]] = []

    for _, quad in scored:
        bbox = _quad_to_bbox(quad)
        if any(_compute_iou(bbox, b) > iou_threshold for b in selected_boxes):
            continue
        selected.append(quad)
        selected_boxes.append(bbox)
        if len(selected) >= max_candidates:
            break

    if not selected and scored:
        selected.append(order_points(scored[0][1]))

    return selected


def detect_plate(image: np.ndarray) -> Optional[np.ndarray]:
    candidates = detect_plate_candidates(image, max_candidates=1)
    if not candidates:
        return None
    return candidates[0]


def crop_plate(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = order_points(quad)

    wA = np.linalg.norm(quad[2] - quad[3])
    wB = np.linalg.norm(quad[1] - quad[0])
    hA = np.linalg.norm(quad[1] - quad[2])
    hB = np.linalg.norm(quad[0] - quad[3])

    width = int(max(wA, wB))
    height = int(max(hA, hB))
    width = max(width, 2)
    height = max(height, 2)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (width, height))


def detect_and_crop(image: np.ndarray) -> Optional[np.ndarray]:
    quad = detect_plate(image)
    if quad is None:
        return None
    return crop_plate(image, quad)


def detect_and_crop_candidates(
    image: np.ndarray, max_candidates: int = 5, iou_threshold: float = 0.25
) -> List[np.ndarray]:
    quads = detect_plate_candidates(image, max_candidates=max_candidates, iou_threshold=iou_threshold)
    crops: List[np.ndarray] = []
    for quad in quads:
        crop = crop_plate(image, quad)
        if crop is not None and crop.size > 0:
            crops.append(crop)
    return crops
