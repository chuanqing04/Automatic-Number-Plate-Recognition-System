
# License Plate Recognition & State Identification System

## Overview
A PySide6 GUI application wrapping an image processing pipeline for license plate recognition and Malaysian state identification.

## Directory Structure
- `1Preprocess/`: Image preprocessing logic.
- `2TextExtract/`: OCR logic (PaddleOCR wrapper).
- `3StateIdentify/`: State identification logic.
- `4GUI/`: MVC GUI Application (Main Window, Controller, Model).

## Requirements
- Python 3.8+
- PySide6
- PaddleOCR
- OpenCV (opencv-python)
- NumPy

## How to Run
1. Navigate to the `4GUI` directory or Root.
2. Run the main entry point:
   ```bash
   python 4GUI/main.py
   ```
   (Or `cd 4GUI && python main.py`)

## Usage
1. Click **Load Image** to select an input image.
2. Click **Run Preprocess**.
3. Click **Run OCR**.
4. Click **Run State Identify**.
5. Alternatively, click **Run Full Pipeline** to execute all steps sequentially.
