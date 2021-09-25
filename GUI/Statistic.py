from PyQt5 import QtGui
from GUI.Ui_Statistic import Ui_Statistic
from PyQt5.QtWidgets import QFrame
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import os
import shutil
from SQL_Connection.SQLConnection import SQLConnection

class Statistic(QFrame, Ui_Statistic):
    def __init__(self, parent = None):
        QFrame.__init__(self, parent=parent)
        self.setupUi(self)

        self.ID_Vehicle = None
        self.file_path = None

        self.loadData()

        # Event
        self.dataTab.cellClicked.connect(self.cellClickOnDataTable)
        self.btDel.clicked.connect(self.deleteOnDataTable)
        self.btDelAll.clicked.connect(self.deleteAllOnDataTable)

    def alert(self, title, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.exec_()
    
    def loadData(self):
        SQL = SQLConnection()
        dataResults = SQL.queryData("Select ID_Vehicle, Speed, Violation_Error, Time From ViolatingVehicle")
        self.dataTab.setRowCount(0)
        for row_number, row_data in enumerate(dataResults):
            self.dataTab.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.dataTab.setItem(row_number, column_number, QtWidgets.QTableWidgetItem(str(data)))
        
        self.dataTab.resizeColumnsToContents()

    def cellClickOnDataTable(self):
        row = self.dataTab.currentRow()
        item = self.dataTab.item(row, 0)
        self.ID_Vehicle = item.text()
        self.lbID.setText(self.ID_Vehicle)
        SQL = SQLConnection()
        dataImg = SQL.queryDataOnly1("Select File_Path From ViolatingVehicle Where ID_Vehicle = '{}'".format(item.text()))
        item = self.dataTab.item(row, 1)
        self.lbViolation.setText(item.text())
        item = self.dataTab.item(row, 2)
        self.lbTime.setText(item.text())
        self.lbImg.setText(str(dataImg[0]))
        self.file_path = "violating_vehicle/" + str(dataImg[0])
        img_car = cv2.imread(self.file_path)
        img_car = cv2.resize(img_car, (551, 351))
        rgb_img = cv2.cvtColor(img_car, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgb_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pImage = convertToQtFormat.scaled(551, 351, Qt.KeepAspectRatio)
        self.lbImgVeh.setPixmap(QtGui.QPixmap(pImage))

    def deleteOnDataTable(self):
        try:
            SQL = SQLConnection()
            SQL.queryNoReturn("Delete From ViolatingVehicle Where ID_Vehicle = {}".format(self.ID_Vehicle))
            self.ID_Vehicle = None
            os.remove(self.file_path)
            self.file_path = None
            self.lbID.setText("")
            self.lbViolation.setText("")
            self.lbTime.setText("")
            self.lbImg.setText("")
            self.lbImgVeh.setPixmap(QPixmap("GUI/ImageGUI/imgNone.png"))
            self.alert(title="Thông báo", message="Thao tác thực hiện thành công")
        except:
            self.alert(title="Cảnh báo", message="Thao tác thực hiện thất bại")
        finally:
            self.loadData()


    def deleteAllOnDataTable(self):
        dir_path = "violating_vehicle"
        try:
            SQL = SQLConnection()
            SQL.queryNoReturn("Delete From ViolatingVehicle")
            for file in os.listdir(dir_path):
                if file[-3:] in ["jpg"]:
                    os.remove(dir_path + "/" + file)
                else:
                    pass
            self.lbID.setText("")
            self.lbViolation.setText("")
            self.lbTime.setText("")
            self.lbImg.setText("")
            self.lbImgVeh.setPixmap(QPixmap("GUI/ImageGUI/imgNone.png"))
            self.alert(title="Thông báo", message="Thao tác thực hiện thành công!")
        except:
            self.alert(title="Cảnh báo", message="Thao tác thực hiện thất bại!")
        finally:
            self.loadData()
        