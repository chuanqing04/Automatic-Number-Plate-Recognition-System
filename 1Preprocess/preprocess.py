
import cv2
import os
import sys
import numpy as np
from typing import Tuple

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

def preprocess(image_path: str) -> str:
    """
    Reads an image, applies preprocessing, and returns the path to the processed image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Standard Preprocessing
    # Grayscale
    gray = PlatePreprocessor.to_grayscale(image)
    
    # Return gray image as 'processed'
    processed_image = gray

    # Save output to this folder
    output_filename = "processed_latest.png"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    
    cv2.imwrite(output_path, processed_image)
    
    return output_path
