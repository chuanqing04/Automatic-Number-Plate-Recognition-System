
import sys
import os
import re
import cv2

# Add sibling directories to path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

preprocess_dir = os.path.join(project_root, '1Preprocess')
detect_dir = os.path.join(project_root, '2DetectSegment')
extract_dir = os.path.join(project_root, '3TextExtract')
identify_dir = os.path.join(project_root, '4StateIdentify')

for p in [detect_dir, preprocess_dir, extract_dir, identify_dir]:
    if p not in sys.path:
        sys.path.append(p)

# Lazy import engine to avoid slow startup for just import check
# But Model is instantiated on startup
try:
    from preprocess import preprocess
    from extract import extract_text, OCREngine # Direct import for initialization
    from identify import identify_state, write_state_json
    from DetectSegment import detect_and_crop, detect_and_crop_candidates
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")

class IPPRModel:
    def __init__(self):
        self.input_image_path = None
        self.processed_image_path = None
        self.ocr_result = None
        self.state_result = None
        self._plate_candidates = []
        self._candidate_index = 0
        self._max_attempts = 20
        
        # Initialize Engine once
        self.engine = None
        self.engine = OCREngine(use_gpu=False)
        print("OCR Engine Initialized.")

    def set_image_path(self, path):
        self.input_image_path = path
        self.processed_image_path = None
        self.ocr_result = None
        self.state_result = None
        self._plate_candidates = []
        self._candidate_index = 0
        self._max_attempts = 20

    @staticmethod
    def _matches_plate_pattern(text: str) -> bool:
        clean = re.sub(r'[^A-Z0-9]', '', (text or '').upper())
        return re.fullmatch(r'[A-Z]{2,3}[0-9]{2,4}', clean) is not None

    def run_preprocess(self):
        if not self.input_image_path:
            raise ValueError("No input image loaded.")

        preprocessed_path = preprocess(self.input_image_path)
        img = cv2.imread(preprocessed_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Cannot read preprocessed image.")

        plate_crops = detect_and_crop_candidates(
            img, max_candidates=self._max_attempts, iou_threshold=0.25
        )
        if not plate_crops:
            raise ValueError("Plate not detected")

        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        self._plate_candidates = []
        for idx, crop in enumerate(plate_crops):
            candidate_path = os.path.join(temp_dir, f"cropped_plate_{idx}.png")
            cv2.imwrite(candidate_path, crop)
            self._plate_candidates.append(candidate_path)

        self._candidate_index = 0

        # Preprocess the top-ranked candidate so the UI shows something immediately
        self.processed_image_path = preprocess(self._plate_candidates[0])
        return self.processed_image_path


    def run_ocr(self):
        if not self._plate_candidates:
            raise ValueError("Preprocessing must be run before OCR.")
        
        if not self.engine:
             raise RuntimeError("OCR Engine not initialized.")

        results = []
        attempts = 0
        while (
            self._candidate_index < len(self._plate_candidates)
            and attempts < self._max_attempts
        ):
            candidate_path = self._plate_candidates[self._candidate_index]

            self.processed_image_path = candidate_path
            result = extract_text(candidate_path, engine=self.engine)
            plate_text = result.get("plate_text", "")
            valid_pattern = self._matches_plate_pattern(plate_text)

            result["pattern_valid"] = valid_pattern
            result["candidate_index"] = self._candidate_index
            result["processed_path"] = self.processed_image_path

            results.append((self._candidate_index, candidate_path, plate_text, result))
            self.ocr_result = result

            self._candidate_index += 1
            attempts += 1

        if not results:
            raise ValueError("OCR failed to produce any result.")

        def score_item(item):
            idx, _, _, res = item
            conf = float(res.get("confidence", 0.0))
            pattern = 1 if res.get("pattern_valid") else 0
            return (pattern, conf, -idx)  # prefer pattern match, then confidence, then earlier idx

        best_idx, best_path, best_text, best_result = max(results, key=score_item)
        self.processed_image_path = best_result.get("processed_path", best_path)
        self.ocr_result = best_result
        return best_result

    def run_identify(self):
        if not self.ocr_result:
            raise ValueError("OCR must be run before State Identification.")

        self.state_result = identify_state(self.ocr_result)

        out_dir = os.path.join(project_root, "output")
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "state_result.json")

        write_state_json(self.ocr_result, out_json)

        return self.state_result

    def run_full_pipeline(self):
        self.run_preprocess()
        self.run_ocr()
        self.run_identify()
        return self.state_result
