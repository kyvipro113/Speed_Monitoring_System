﻿# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UiStatistic.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Statistic(object):
    def setupUi(self, Statistic):
        Statistic.setObjectName("Statistic")
        Statistic.resize(1191, 531)
        self.label_7 = QtWidgets.QLabel(Statistic)
        self.label_7.setGeometry(QtCore.QRect(310, 0, 531, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.dataTab = QtWidgets.QTableWidget(Statistic)
        self.dataTab.setGeometry(QtCore.QRect(10, 60, 591, 461))
        self.dataTab.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.dataTab.setObjectName("dataTab")
        self.dataTab.setColumnCount(4)
        self.dataTab.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        item.setFont(font)
        self.dataTab.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        item.setFont(font)
        self.dataTab.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        item.setFont(font)
        self.dataTab.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        item.setFont(font)
        self.dataTab.setHorizontalHeaderItem(3, item)
        self.dataTab.horizontalHeader().setDefaultSectionSize(140)
        self.dataTab.horizontalHeader().setMinimumSectionSize(37)
        self.lbImgVeh = QtWidgets.QLabel(Statistic)
        self.lbImgVeh.setGeometry(QtCore.QRect(630, 60, 551, 351))
        self.lbImgVeh.setStyleSheet("background-color:rgb(34, 177, 76)")
        self.lbImgVeh.setText("")
        self.lbImgVeh.setObjectName("lbImgVeh")
        self.verticalLayoutWidget = QtWidgets.QWidget(Statistic)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(640, 421, 108, 108))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Statistic)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(750, 420, 301, 108))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lbID = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbID.setFont(font)
        self.lbID.setObjectName("lbID")
        self.verticalLayout_2.addWidget(self.lbID)
        self.lbViolation = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbViolation.setFont(font)
        self.lbViolation.setObjectName("lbViolation")
        self.verticalLayout_2.addWidget(self.lbViolation)
        self.lbTime = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbTime.setFont(font)
        self.lbTime.setObjectName("lbTime")
        self.verticalLayout_2.addWidget(self.lbTime)
        self.lbImg = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbImg.setFont(font)
        self.lbImg.setObjectName("lbImg")
        self.verticalLayout_2.addWidget(self.lbImg)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Statistic)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(1070, 430, 102, 80))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btDel = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.btDel.setFont(font)
        self.btDel.setObjectName("btDel")
        self.verticalLayout_3.addWidget(self.btDel)
        self.btDelAll = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.btDelAll.setFont(font)
        self.btDelAll.setObjectName("btDelAll")
        self.verticalLayout_3.addWidget(self.btDelAll)

        self.retranslateUi(Statistic)
        QtCore.QMetaObject.connectSlotsByName(Statistic)

    def retranslateUi(self, Statistic):
        _translate = QtCore.QCoreApplication.translate
        Statistic.setWindowTitle(_translate("Statistic", "Frame"))
        self.label_7.setText(_translate("Statistic", "Thống Kê Phương Tiện Giao Thông Vi Phạm"))
        item = self.dataTab.horizontalHeaderItem(0)
        item.setText(_translate("Statistic", "Mã Phương Tiện"))
        item = self.dataTab.horizontalHeaderItem(1)
        item.setText(_translate("Statistic", "Tốc Độ"))
        item = self.dataTab.horizontalHeaderItem(2)
        item.setText(_translate("Statistic", "Lỗi Vi Phạm"))
        item = self.dataTab.horizontalHeaderItem(3)
        item.setText(_translate("Statistic", "Thời Gian"))
        self.label_2.setText(_translate("Statistic", "Mã P.Tiện:"))
        self.label_4.setText(_translate("Statistic", "Lỗi Vi Phạm:"))
        self.label_3.setText(_translate("Statistic", "Thời Gian:"))
        self.label_8.setText(_translate("Statistic", "Ảnh P.Tiện:"))
        self.lbID.setText(_translate("Statistic", "ID"))
        self.lbViolation.setText(_translate("Statistic", "vio"))
        self.lbTime.setText(_translate("Statistic", "yyyy-mm-dd hh:mm:ss"))
        self.lbImg.setText(_translate("Statistic", "img"))
        self.btDel.setText(_translate("Statistic", "Xóa"))
        self.btDelAll.setText(_translate("Statistic", "Xóa Tất Cả"))


