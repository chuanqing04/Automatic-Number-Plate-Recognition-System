
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QTextEdit, QGroupBox, QFrame,
                               QFormLayout, QProgressBar, QApplication)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("License Plate Recognition & State Identification System")
        self.resize(1000, 700)
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Layout (Vertical: Top Title, Middle Content, Bottom Controls, Status)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 1. Top Section - Title
        self.title_label = QLabel("License Plate Recognition & State Identification System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(self.title_label)
        
        # 2. Middle Section - Splitter for Image and Results
        self.middle_layout = QHBoxLayout()
        self.main_layout.addLayout(self.middle_layout)
        
        # Left Panel - Image Display
        self.left_panel = QFrame()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Input Image Viewer
        self.input_image_label = QLabel()
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.input_image_label.setMinimumSize(400, 300)
        self.input_image_caption = QLabel("Input Image")
        self.input_image_caption.setAlignment(Qt.AlignCenter)
        
        self.left_layout.addWidget(self.input_image_label)
        self.left_layout.addWidget(self.input_image_caption)
        
        # Plate Region Viewer
        self.plate_region_label = QLabel()
        self.plate_region_label.setAlignment(Qt.AlignCenter)
        self.plate_region_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.plate_region_label.setMinimumSize(200, 100)
        self.plate_region_caption = QLabel("Detected Plate")
        self.plate_region_caption.setAlignment(Qt.AlignCenter)
        
        self.left_layout.addWidget(self.plate_region_label)
        self.left_layout.addWidget(self.plate_region_caption)
        
        self.middle_layout.addWidget(self.left_panel, stretch=2)
        
        # Right Panel - Results (Structured UI)
        self.right_panel = QFrame()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # JSON Output Section (Converted to UI)
        self.group_ocr = QGroupBox("Extraction Result")
        self.form_ocr = QFormLayout()
        
        self.lbl_plate_text = QLabel("N/A")
        self.lbl_plate_text.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.lbl_confidence = QLabel("N/A")
        self.lbl_method = QLabel("N/A")
        
        self.form_ocr.addRow("Plate Text:", self.lbl_plate_text)
        self.form_ocr.addRow("Confidence:", self.lbl_confidence)
        self.form_ocr.addRow("Method:", self.lbl_method)
        
        self.group_ocr.setLayout(self.form_ocr)
        
        # State ID Section
        self.group_state = QGroupBox("State Identification Result")
        self.form_state = QFormLayout()
        
        self.lbl_state = QLabel("N/A")
        self.lbl_state.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
        
        self.form_state.addRow("Identified State:", self.lbl_state)
        
        self.group_state.setLayout(self.form_state)
        
        self.right_layout.addWidget(self.group_ocr)
        self.right_layout.addWidget(self.group_state)
        self.right_layout.addStretch() # Push up
        
        self.middle_layout.addWidget(self.right_panel, stretch=1)
        
        # 3. Bottom Control Panel
        self.control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load Image")
        self.btn_preprocess = QPushButton("Run Preprocess")
        self.btn_ocr = QPushButton("Run OCR")
        self.btn_identify = QPushButton("Run State Identify")
        self.btn_full = QPushButton("Run Full Pipeline")
        
        self.control_layout.addWidget(self.btn_load)
        self.control_layout.addWidget(self.btn_preprocess)
        self.control_layout.addWidget(self.btn_ocr)
        self.control_layout.addWidget(self.btn_identify)
        self.control_layout.addWidget(self.btn_full)
        
        self.main_layout.addLayout(self.control_layout)
        
        
        # 4. Status Panel
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.main_layout.addWidget(self.progress_bar)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(100)
        self.main_layout.addWidget(self.status_log)
        
        # Logic to enable/disable buttons
        self.btn_preprocess.setEnabled(False)
        self.btn_ocr.setEnabled(False)
        self.btn_identify.setEnabled(False)
        self.btn_full.setEnabled(False)

    def log(self, message):
        self.status_log.append(message)

    def clear_log(self):
        self.status_log.clear()
        
    def set_loading(self, loading=True, message=None):
        if loading:
            self.progress_bar.show()
            if message:
                self.log(message)
            # Force UI update so bar appears before blocking operation
            QApplication.processEvents()
        else:
            self.progress_bar.hide()

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.input_image_label.setPixmap(pixmap.scaled(
                self.input_image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        else:
            self.log(f"Error: Could not display image {image_path}")

    def display_plate(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.plate_region_label.setPixmap(pixmap.scaled(
                self.plate_region_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
    
    def update_ocr_result(self, text, conf, method="Standard"):
        self.lbl_plate_text.setText(str(text))
        self.lbl_confidence.setText(f"{float(conf):.2%}" if isinstance(conf, (float, int)) else str(conf))
        self.lbl_method.setText(str(method))

    def update_state_result(self, state):
        self.lbl_state.setText(str(state))
