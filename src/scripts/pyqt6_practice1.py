# from PyQt6.QtWidgets import QApplication, QWidget, QPushButton
from PyQt6.QtWidgets import *
import sys

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("My App")
        
        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)
        button.clicked.connect(self.the_button_was_toggled)
        
        self.setCentralWidget(button)
        # self.setFixedSize(400,300) # Sets a fixed "width x height" for the window
        # self.setMinimumSize(150, 100) # Sets a "minimum size" for the window
        # self.setMaximumSize(940, 700) # Sets a "maximum size" for the window
        
    def the_button_was_clicked(self):
        print("Clicked!")
        
    def the_button_was_toggled(self, checked):
        print(f"Checked? {checked}")
        
app = QApplication(sys.argv)
    
window = MainWindow()
window.show()

app.exec()

# # Instantiates the app
# app = QApplication(sys.argv)

# # Creates a Qt widget (window)
# window1 = QWidget()
# window1.show() # Windows are hidden by default

# # Creates a window with a push button, as any QtWidget can be a window
# window2 = QPushButton("Push Me")
# window2.show()

# window3 = QMainWindow()
# window3.show()

# app.exec()