# main.py

"""
This is the main entry point of the application.

Data Sent:
- Initializes the QApplication and starts the main event loop.

Data Returned:
- N/A
"""

import sys
from PyQt5.QtWidgets import QApplication
from main_app import GazeApp

def main():
    app = QApplication(sys.argv)
    gaze_app = GazeApp()
    gaze_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
