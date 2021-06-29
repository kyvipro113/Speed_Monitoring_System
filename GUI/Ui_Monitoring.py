# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UiMonitoring.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Monitoring(object):
    def setupUi(self, Monitoring):
        Monitoring.setObjectName("Monitoring")
        Monitoring.resize(1191, 531)
        self.lbImg = QtWidgets.QLabel(Monitoring)
        self.lbImg.setGeometry(QtCore.QRect(10, 40, 850, 480))
        self.lbImg.setStyleSheet("background-color:rgb(34, 177, 76)")
        self.lbImg.setText("")
        self.lbImg.setObjectName("lbImg")
        self.label_7 = QtWidgets.QLabel(Monitoring)
        self.label_7.setGeometry(QtCore.QRect(420, 0, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(950, 170, 131, 51))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btChooseVideo = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.btChooseVideo.setFont(font)
        self.btChooseVideo.setObjectName("btChooseVideo")
        self.horizontalLayout_3.addWidget(self.btChooseVideo)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(880, 130, 131, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(1010, 130, 171, 41))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lbNameVideo = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.lbNameVideo.setFont(font)
        self.lbNameVideo.setObjectName("lbNameVideo")
        self.horizontalLayout_2.addWidget(self.lbNameVideo)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(1010, 80, 171, 41))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.comboInput = QtWidgets.QComboBox(self.horizontalLayoutWidget_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.comboInput.setFont(font)
        self.comboInput.setObjectName("comboInput")
        self.horizontalLayout_4.addWidget(self.comboInput)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(880, 80, 131, 41))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.verticalLayoutWidget = QtWidgets.QWidget(Monitoring)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(950, 310, 131, 121))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btStart = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.btStart.setFont(font)
        self.btStart.setObjectName("btStart")
        self.verticalLayout.addWidget(self.btStart)
        self.btEnd = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.btEnd.setFont(font)
        self.btEnd.setObjectName("btEnd")
        self.verticalLayout.addWidget(self.btEnd)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(1010, 220, 171, 41))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.txtPPM = QtWidgets.QLineEdit(self.horizontalLayoutWidget_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.txtPPM.setFont(font)
        self.txtPPM.setObjectName("txtPPM")
        self.horizontalLayout_6.addWidget(self.txtPPM)
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(Monitoring)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(880, 220, 131, 41))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_6 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_7.addWidget(self.label_6)

        self.retranslateUi(Monitoring)
        QtCore.QMetaObject.connectSlotsByName(Monitoring)

    def retranslateUi(self, Monitoring):
        _translate = QtCore.QCoreApplication.translate
        Monitoring.setWindowTitle(_translate("Monitoring", "Frame"))
        self.label_7.setText(_translate("Monitoring", "Vehicle Speed Monitoring"))
        self.btChooseVideo.setText(_translate("Monitoring", "Choose Video"))
        self.label_2.setText(_translate("Monitoring", "Input Video:"))
        self.lbNameVideo.setText(_translate("Monitoring", "Video.mp4"))
        self.label_5.setText(_translate("Monitoring", "Input Type:"))
        self.btStart.setText(_translate("Monitoring", "Start"))
        self.btEnd.setText(_translate("Monitoring", "End"))
        self.label_6.setText(_translate("Monitoring", "Pixel Per Meter:"))
