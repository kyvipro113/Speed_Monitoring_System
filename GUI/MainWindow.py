from PyQt5.QtWidgets import QMainWindow
from GUI.Ui_MainWindow import Ui_MainWindow
from GUI.Monitoring import Monitoring
from GUI.Statistic import Statistic

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent=parent)
        self.setupUi(self)

        # User Control
        self.UIMonitoring = None
        self.UIStatistic = None

        #Event
        self.tabMonitoring.tabBarClicked.connect(self.tabClicked)

    def cleanUI(self):
        if(self.UIMonitoring):
            self.UIMonitoring.hide()
            del self.UIMonitoring
            self.UIMonitoring = None
        if(self.UIStatistic):
            self.UIStatistic.hide()
            del self.UIStatistic
            self.UIStatistic = None

    def tabClicked(self, index):
        self.cleanUI()
        if(index == 0):
            self.UIMonitoring = Monitoring(self.mainFrame)
            self.UIMonitoring.show()
        if(index == 1):
            self.UIStatistic = Statistic(self.mainFrame)
            self.UIStatistic.show()
