
import sys
from PySide6.QtWidgets import QApplication
from model import IPPRModel
from view import MainWindow
from controller import Controller

def main():
    app = QApplication(sys.argv)
    
    # Instantiate MVC components
    model = IPPRModel()
    view = MainWindow()
    controller = Controller(model, view)
    
    view.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
