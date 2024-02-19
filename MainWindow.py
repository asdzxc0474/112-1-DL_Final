from UI.MainWindow_UI import Ui_MainWindow
from UI.ResNet50_UI import ResNet50_Ui_Form
from UI.MobileNetV3_UI import MobileNetV3_Ui_Form
from dataloader import random_load_testdata
from load_model import load_renet50, load_mobilenetv3, load_vgg19
from UI.VGG19_UI import VGG19_Ui_Form
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        MainWindow = Ui_MainWindow()
        MainWindow.setupUi(self)
        MainWindow.pushButton.clicked.connect(self.VGG19_event)
        MainWindow.pushButton_2.clicked.connect(self.ResNet50_event)
        MainWindow.pushButton_3.clicked.connect(self.MobileNetV3_event)
        MainWindow.pushButton_4.clicked.connect(self.close)
    def VGG19_event(self):
        widget.setCurrentWidget(VGG19_Window)
        print("Model:VGG19")
    
    def ResNet50_event(self):
        widget.setCurrentWidget(resnt_window)
        print("Model:ResNet50")

    def MobileNetV3_event(self):
        widget.setCurrentWidget(MobileNet_window)
        print("Model:MobileNetV3_Large")
    
    def close(self):
        qApp = QApplication.instance()
        qApp.quit()
class ResNet(QtWidgets.QMainWindow):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet_window = ResNet50_Ui_Form()
        self.resnet_window.setupUi(self)
        self.resnet_window.pushButton.clicked.connect(self.trainLoss_AC)
        self.resnet_window.pushButton_2.clicked.connect(self.ValLoss_AC)
        self.resnet_window.pushButton_3.clicked.connect(self.Confusion_Matrix)
        self.resnet_window.pushButton_4.clicked.connect(self.Test_Demo)
        self.resnet_window.pushButton_5.clicked.connect(self.close)
        self.resnet_window.pushButton_6.clicked.connect(self.previous)
        self.resnet_window.pushButton_7.clicked.connect(self.next)
        self.resnet_window.pushButton_6.setDisabled(True)
        self.resnet_window.pushButton_7.setDisabled(True)
        self.model = load_renet50()
        self.data = random_load_testdata('./split_dataset/test')
        self.datalist = []
        self.flag = 0
    
    def trainLoss_AC(self):
        self.resnet_window.pushButton_4.setDisabled(False)
        self.resnet_window.pushButton_6.setDisabled(True)
        self.resnet_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/resnet50_Train_accuracy_loss.jpg')
        self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(img)) 
        print("trainLoss_AC")
    
    def ValLoss_AC(self):
        self.resnet_window.pushButton_4.setDisabled(False)
        self.resnet_window.pushButton_6.setDisabled(True)
        self.resnet_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/resnet50_Val_accuracy_loss.jpg')
        self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(img)) 
        print("ValLoss_AC")

    def Confusion_Matrix(self):
        self.resnet_window.pushButton_4.setDisabled(False)
        self.resnet_window.pushButton_6.setDisabled(True)
        self.resnet_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/resnet50_confusion_matrix.png')
        scaled_img = img.scaled(self.resnet_window.label.size(), QtCore.Qt.KeepAspectRatio)
        self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))    

    def Test_Demo(self):
        self.resnet_window.pushButton_4.setDisabled(True)
        self.resnet_window.pushButton_6.setDisabled(False)
        self.resnet_window.pushButton_7.setDisabled(False)
        try:
            data_path = self.datalist[self.flag]
        except:
            data_path = self.data.ramdom_out()    
            self.datalist.append(data_path)
        img2 = self.model.ret(data_path)
        scaled_img = img2.scaled(self.resnet_window.label.size(), QtCore.Qt.KeepAspectRatio)
        self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
    def previous(self):
        if(self.flag > 0):
            self.flag = self.flag -1
            data =  self.datalist[self.flag]
            img2= self.model.ret(data)
            scaled_img = img2.scaled(self.resnet_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
        else:
            pass
    def next(self):
        try:
            self.flag = self.flag + 1
            data =  self.datalist[self.flag]
            img2= self.model.ret(data)
            scaled_img = img2.scaled(self.resnet_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
        except:
            data_path = self.data.ramdom_out()
            img2= self.model.ret(data_path)
            scaled_img = img2.scaled(self.resnet_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.resnet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
            self.datalist.append(data_path)

    def close(self):
        widget.setCurrentWidget(main_window)
class VGG19(QtWidgets.QMainWindow):
    def __init__(self):
        super(VGG19, self).__init__()
        self.VGG_window = VGG19_Ui_Form()
        self.VGG_window.setupUi(self)
        self.VGG_window.pushButton.clicked.connect(self.trainLoss_AC)
        self.VGG_window.pushButton_2.clicked.connect(self.ValLoss_AC)
        self.VGG_window.pushButton_3.clicked.connect(self.Confusion_Matrix)
        self.VGG_window.pushButton_4.clicked.connect(self.Test_Demo)
        self.VGG_window.pushButton_5.clicked.connect(self.close)
        self.VGG_window.pushButton_6.clicked.connect(self.previous)
        self.VGG_window.pushButton_7.clicked.connect(self.next)
        self.VGG_window.pushButton_6.setDisabled(True)
        self.VGG_window.pushButton_7.setDisabled(True)
        self.data = random_load_testdata('./split_dataset/test')
        self.model = load_vgg19()
        self.datalist = []
        self.flag = 0
    def trainLoss_AC(self):
        self.VGG_window.pushButton_4.setDisabled(False)
        self.VGG_window.pushButton_6.setDisabled(True)
        self.VGG_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/vgg19_Train_accuracy_loss.jpg')                 # 讀取圖片
        self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(img)) 
        print("trainLoss_AC")
    
    def ValLoss_AC(self):
        self.VGG_window.pushButton_4.setDisabled(False)
        self.VGG_window.pushButton_6.setDisabled(True)
        self.VGG_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/vgg19_Val_accuracy_loss.jpg')                 # 讀取圖片
        self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(img)) 
        print("trainLoss_AC")

    def Confusion_Matrix(self):
        self.VGG_window.pushButton_4.setDisabled(False)
        self.VGG_window.pushButton_6.setDisabled(True)
        self.VGG_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/vgg19_confusion_matrix.png')
        scaled_img = img.scaled(self.VGG_window.label.size(), QtCore.Qt.KeepAspectRatio)
        self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))    
        
    def Test_Demo(self):
        self.VGG_window.pushButton_4.setDisabled(True)
        self.VGG_window.pushButton_6.setDisabled(False)
        self.VGG_window.pushButton_7.setDisabled(False)
        try:
            data_path = self.datalist[self.flag]
        except:
            data_path = self.data.ramdom_out()    
            self.datalist.append(data_path)
        img2= self.model.ret(data_path)
        scaled_img = img2.scaled(self.VGG_window.label.size(), QtCore.Qt.KeepAspectRatio)
        self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
    def previous(self):
        if(self.flag > 0):
            self.flag = self.flag -1
            data =  self.datalist[self.flag]
            img2= self.model.ret(data)
            scaled_img = img2.scaled(self.VGG_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
        else:
            pass
    def next(self):
        try:
            self.flag = self.flag + 1
            data =  self.datalist[self.flag]
            img2= self.model.ret(data)
            scaled_img = img2.scaled(self.VGG_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
        except:
            data_path = self.data.ramdom_out()
            img2= self.model.ret(data_path)
            scaled_img = img2.scaled(self.VGG_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.VGG_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
            self.datalist.append(data_path)
    def close(self):
        widget.setCurrentWidget(main_window)    
class MobileNet(QtWidgets.QMainWindow):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.MobileNet_window = MobileNetV3_Ui_Form()
        self.MobileNet_window.setupUi(self)
        self.MobileNet_window.pushButton.clicked.connect(self.trainLoss_AC)
        self.MobileNet_window.pushButton_2.clicked.connect(self.ValLoss_AC)
        self.MobileNet_window.pushButton_3.clicked.connect(self.Confusion_Matrix)
        self.MobileNet_window.pushButton_4.clicked.connect(self.Test_Demo)
        self.MobileNet_window.pushButton_5.clicked.connect(self.close)
        self.MobileNet_window.pushButton_6.clicked.connect(self.previous)
        self.MobileNet_window.pushButton_7.clicked.connect(self.next)
        self.MobileNet_window.pushButton_6.setDisabled(True)
        self.MobileNet_window.pushButton_7.setDisabled(True)
        self.data = random_load_testdata('./split_dataset/test')
        self.model = load_mobilenetv3()
        self.datalist = []
        self.flag = 0
    def trainLoss_AC(self):
        self.MobileNet_window.pushButton_4.setDisabled(False)
        self.MobileNet_window.pushButton_6.setDisabled(True)
        self.MobileNet_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/MobileNetV3_Train_accuracy_loss.jpg')
        self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(img))
        print("trainLoss_AC")
    
    def ValLoss_AC(self):
        self.MobileNet_window.pushButton_4.setDisabled(False)
        self.MobileNet_window.pushButton_6.setDisabled(True)
        self.MobileNet_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/MobileNetV3_Val_accuracy_loss.jpg')
        self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(img))
        print("ValLoss_AC")

    def Confusion_Matrix(self):
        self.MobileNet_window.pushButton_4.setDisabled(False)
        self.MobileNet_window.pushButton_6.setDisabled(True)
        self.MobileNet_window.pushButton_7.setDisabled(True)
        img = QtGui.QImage('fig/mobilenet_v3_confusion_matrix.png')
        scaled_img = img.scaled(self.MobileNet_window.label.size(), QtCore.Qt.KeepAspectRatio)
        self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))

    def Test_Demo(self):
        self.MobileNet_window.pushButton_4.setDisabled(True)
        self.MobileNet_window.pushButton_6.setDisabled(False)
        self.MobileNet_window.pushButton_7.setDisabled(False)       
        try:
            data_path = self.datalist[self.flag]
        except:
            data_path = self.data.ramdom_out()    
            self.datalist.append(data_path)
        img2= self.model.ret(data_path)
        scaled_img = img2.scaled(self.MobileNet_window.label.size(), QtCore.Qt.KeepAspectRatio)
        self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))

    def previous(self):
        if(self.flag > 0):
            self.flag = self.flag -1
            data =  self.datalist[self.flag]
            img2= self.model.ret(data)
            scaled_img = img2.scaled(self.MobileNet_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
        else:
            pass
    def next(self):
        try:
            self.flag = self.flag + 1
            data =  self.datalist[self.flag]        
            img2= self.model.ret(data)
            scaled_img = img2.scaled(self.MobileNet_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
        except:
            data_path = self.data.ramdom_out()
            img2= self.model.ret(data_path)
            scaled_img = img2.scaled(self.MobileNet_window.label.size(), QtCore.Qt.KeepAspectRatio)
            self.MobileNet_window.label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
            self.datalist.append(data_path)

    
    def close(self):
        widget.setCurrentWidget(main_window)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication([])
    widget = QtWidgets.QStackedWidget()

    main_window = Main()
    resnt_window = ResNet()
    VGG19_Window = VGG19()
    MobileNet_window = MobileNet()

    widget.addWidget(main_window)
    widget.addWidget(resnt_window)
    widget.addWidget(VGG19_Window)
    widget.addWidget(MobileNet_window)

    widget.setCurrentWidget(main_window)
    widget.showFullScreen()
    sys.exit(app.exec_())