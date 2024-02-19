import torch
import torchvision.transforms as transforms
import torchvision
import cv2
from dataloader import putext, cv2qimg
class load_renet50:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('weight/resnet50_best.pt')
        print(self.model)
        self.test_transform = transforms.Compose([transforms.Resize((224, 224)),torchvision.transforms.ToTensor(),\
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                        std=[0.229, 0.224, 0.225])])
        self.model.to(self.device)
    def ret(self, imgpath):
        img = cv2.imread(imgpath)
        self.img = transforms.ToPILImage()(img)
        self.img =self.test_transform(self.img)
        self.img = self.img.unsqueeze(0)
        outputs = self.model(self.img.cuda())
        _, self.preds = torch.max(outputs, 1)
        img2 = putext(img, self.preds)
        self.img = cv2qimg(img2)
        return  self.img
class load_mobilenetv3:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('weight/mobilenetv3_best.pt')
        print(self.model)
        self.test_transform = transforms.Compose([transforms.Resize((224, 224)),torchvision.transforms.ToTensor(),\
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                       std=[0.229, 0.224, 0.225])])
        self.model.to(self.device)
        self.model.eval()
    def ret(self, imgpath):
        img = cv2.imread(imgpath)
        self.img = transforms.ToPILImage()(img)
        self.img =self.test_transform(self.img)
        self.img = self.img.unsqueeze(0)
        outputs = self.model(self.img.cuda())
        _, self.preds = torch.max(outputs, 1)
        img2 = putext(img, self.preds)
        self.img = cv2qimg(img2)
        return  self.img
class load_vgg19:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('weight/vgg19_best.pt')
        print(self.model)
        self.test_transform = transforms.Compose([transforms.Resize((224, 224)),torchvision.transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225])])
        self.model.to(self.device)
        self.model.eval()
    def ret(self, imgpath):
        img = cv2.imread(imgpath)
        self.img = transforms.ToPILImage()(img)
        self.img =self.test_transform(self.img)
        self.img = self.img.unsqueeze(0)
        outputs = self.model(self.img.cuda())
        _, self.preds = torch.max(outputs, 1)
        img2 = putext(img, self.preds)
        self.img = cv2qimg(img2)
        return  self.img
if __name__ == "__main__":
    resnet50 = load_renet50()
    #mobilenetv3 = load_mobilenetv3()
    #vgg19 = load_vgg19()