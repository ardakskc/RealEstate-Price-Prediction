import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QMessageBox, QPlainTextEdit, QVBoxLayout, \
    QLineEdit, QListView, QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize
import matplotlib.pyplot as plt
from Linear_Model import LinearFonk
from MultiLinear_Model import multiLinearFonk
from Lasso_Model import  LassoFonk
from Polynomial_Model import polynomialFonk
from DataSet_Update import MIN_KITCHEN,MIN_AREA,MIN_PRICE,REGION_ID,MAX_KITCHEN,MAX_AREA,MAX_PRICE,basla,features_update,dataset_filter,region_filter,DataSetGoster,DataSetGoster2
from Feature_Selection import feature,select_features

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(300, 200))
        self.setWindowTitle("Russia Real Estate")
        self.setStyleSheet("background-color: white;")
        pybutton = QPushButton('Linear Model', self)
        pybutton.clicked.connect(self.LinearModel)  #fonksyion eklediğimiz kısım burası.
        pybutton.resize(300, 96)
        pybutton.move(400, 400)
        pybutton.setStyleSheet("background-color: open gray")
        pybutton.setFont(QFont('Times', 15))

        self.textbox = QLineEdit(self)
        self.textbox.move(800, 70)
        self.textbox.resize(300, 96)
        self.textbox.setText("Russian Real Estate")
        self.textbox.setFont(QFont('Serif',15))
        self.textbox.setAlignment(QtCore.Qt.AlignCenter)

        self.textbox2 = QListWidget(self)
        self.textbox2.move(100,50)
        self.textbox2.resize(300,96)
        self.textbox2.resize(300,700)
        self.textbox2.move(1300,200)

        self.closeButton = QPushButton(self)
        self.closeButton.setText("Quit")  # text
        self.closeButton.setShortcut('Ctrl+D')  # shortcut key
        self.closeButton.clicked.connect(self.close)
        self.closeButton.setToolTip("Close the widget")  # Tool tip
        self.closeButton.resize(300, 96)
        self.closeButton.move(800,800)
        self.closeButton.setStyleSheet("background-color: open gray")
        self.closeButton.setFont(QFont('Times', 15))

        pybutton2 = QPushButton('MultiLinear Model', self)
        pybutton2.clicked.connect(self.multiLinearModel)
        pybutton2.resize(300, 96)
        pybutton2.move(400, 300)
        pybutton2.setStyleSheet("background-color: open gray")
        pybutton2.setFont(QFont('Times', 15))

        pybutton3 = QPushButton('Lasso Model', self)
        pybutton3.clicked.connect(self.lassoModel)
        pybutton3.resize(300, 96)
        pybutton3.move(400,500)
        pybutton3.setStyleSheet("background-color: open gray")
        pybutton3.setFont(QFont('Times', 15))

        pybutton4 = QPushButton('Polynomial Model', self)
        pybutton4.clicked.connect(self.polynomialModel)
        pybutton4.resize(300, 96)
        pybutton4.move(400, 600)
        pybutton4.setStyleSheet("background-color: open gray")
        pybutton4.setFont(QFont('Times', 15))

        pybutton5 = QPushButton('Feature Selection', self)
        pybutton5.clicked.connect(self.featureYap)
        pybutton5.resize(300, 96)
        pybutton5.move(800,200)
        pybutton5.setStyleSheet("background-color: open gray")
        pybutton5.setFont(QFont('Times', 15))

        pybutton9 = QPushButton('Years-million rubble', self)
        pybutton9.clicked.connect(self.datasetgoster)
        pybutton9.resize(300, 64)
        pybutton9.move(800, 400)
        pybutton9.setStyleSheet("background-color: open gray")
        pybutton9.setFont(QFont('Times', 15))

        pybutton9 = QPushButton('Correalation matrix', self)
        pybutton9.clicked.connect(self.datasetgoster2)
        pybutton9.resize(300, 64)
        pybutton9.move(800, 500)
        pybutton9.setStyleSheet("background-color: open gray")
        pybutton9.setFont(QFont('Times', 15))

        self.data = "test"
        self.degerler2 = 0

    def clickMethod(self):
        QMessageBox.about(self, "Values", "RMSE: " + str(self.degerler2[0]) + "\nMAE: " + str(self.degerler2[1]) + "\nR2: " + str(self.degerler2[2]))
    def clickMethod2(self):
        col = list(self.data.columns.values)
        strdegerler = []

        for i in range(len(self.degerler2)):
            strdegerler.append(col[i+1] + " : " + str(self.degerler2[i]))
        self.textbox2.addItems(strdegerler)

            # burda degerleri string yapıp 1 tane qmessagebox ile popup yaptırcan.

    def LinearModel(self):
        LinearFonk()

    def multiLinearModel(self):
        self.degerler2 = multiLinearFonk()
        self.clickMethod()
        # MainWindow.b.insertPlainText(degerler2)

    def lassoModel(self):
        self.degerler2 = LassoFonk()
        self.clickMethod()

    def polynomialModel(self):
        self.degerler2 = polynomialFonk()
        self.clickMethod()

    def datasetgoster(self):
        DataSetGoster(self.data)
    def datasetgoster2(self):
        DataSetGoster2(self.data)

    def featureYap(self):
        self.degerler2=feature()
        self.clickMethod2()


if __name__ == "__main__": #garantiliyoruz maini

    df = basla()
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.data = df
    mainWin.show()
    sys.exit( app.exec_() )