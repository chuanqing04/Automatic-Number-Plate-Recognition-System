
import json
from PySide6.QtWidgets import QFileDialog, QMessageBox

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
        # Connect signals
        self.view.btn_load.clicked.connect(self.load_image)
        self.view.btn_preprocess.clicked.connect(self.run_preprocess)
        self.view.btn_ocr.clicked.connect(self.run_ocr)
        self.view.btn_identify.clicked.connect(self.run_identify)
        self.view.btn_full.clicked.connect(self.run_full_pipeline)

    def load_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self.view, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if image_path:
            self.model.set_image_path(image_path)
            
            # Reset UI
            self.view.clear_log()
            self.view.plate_region_label.clear()
            self.view.update_ocr_result("N/A", "N/A", "N/A")
            self.view.update_state_result("N/A")

            self.view.display_image(image_path)
            self.view.log(f"Image loaded: {image_path}")
            
            # Enable next step
            self.view.btn_preprocess.setEnabled(True)
            self.view.btn_full.setEnabled(True)
            # Disable subsequent steps
            self.view.btn_ocr.setEnabled(False)
            self.view.btn_identify.setEnabled(False)

    def run_preprocess(self):
        self.view.set_loading(True, "Preprocessing Started...")
        try:
            processed_path = self.model.run_preprocess()
            self.view.log("Preprocess completed.")
            # Display result in plate region viewer
            self.view.display_plate(processed_path)
            
            self.view.btn_ocr.setEnabled(True)
        except Exception as e:
            self.handle_error(f"Preprocessing failed: {e}")
        finally:
            self.view.set_loading(False)

    def run_ocr(self):
        self.view.set_loading(True, "Running OCR (This may take a moment)...")
        try:
            result = self.model.run_ocr()
            self.view.log("OCR completed.")
            
            # Update UI with structured data
            text = result.get("plate_text", "Unknown")
            conf = result.get("confidence", 0.0)
            method = result.get("method", "Standard")
            
            self.view.update_ocr_result(text, conf, method)
            if self.model.processed_image_path:
                # Show the actual crop used for the final OCR result
                self.view.display_plate(self.model.processed_image_path)
            
            self.view.btn_identify.setEnabled(True)
        except Exception as e:
            self.handle_error(f"OCR failed: {e}")
        finally:
            self.view.set_loading(False)

    def run_identify(self):
        self.view.set_loading(True, "Identifying State...")
        try:
            result = self.model.run_identify()
            self.view.log("State identified.")
            
            state = result.get("state", "Unknown")
            self.view.update_state_result(state)
        except Exception as e:
            self.handle_error(f"State ID failed: {e}")
        finally:
            self.view.set_loading(False)

    def run_full_pipeline(self):
        self.view.set_loading(True, "Running Full Pipeline...")
        try:
            # Step 1
            if not self.model.input_image_path:
                self.handle_error("No image loaded.")
                return
            
            # Note: We don't call the other run_ methods to avoid multiple toggling of loading bar
            # We call model methods directly
            self.view.log("1. Preprocessing...")
            processed_path = self.model.run_preprocess()
            self.view.display_plate(processed_path)
            
            # Force update between steps
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Step 2
            self.view.log("2. Running OCR...")
            result = self.model.run_ocr()
            self.view.update_ocr_result(result.get("plate_text"), result.get("confidence"), result.get("method"))
            if self.model.processed_image_path:
                self.view.display_plate(self.model.processed_image_path)
            
            QApplication.processEvents()

            # Step 3
            self.view.log("3. Identifying State...")
            state_res = self.model.run_identify()
            self.view.update_state_result(state_res.get("state"))
            
            self.view.log("Pipeline Completed Successfully.")
            
        except Exception as e:
            self.handle_error(f"Pipeline failed: {e}")
        finally:
            self.view.set_loading(False)

    def handle_error(self, message):
        self.view.log(f"ERROR: {message}")
        QMessageBox.critical(self.view, "Error", message)
