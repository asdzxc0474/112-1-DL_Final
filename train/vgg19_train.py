import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import confusion_matrix

# Data augmentation and normalization for training
# Just normalization for validation
class train():
    def __init__(self, model, num_class):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_transforms = self.data_transform()
        
        number_features = self.model.classifier[6].in_features 
        features = list(self.model.classifier.children())[:-1] # 移除最后一层
        features.extend([torch.nn.Linear(number_features, num_class)]) 
        self.model.classifier = torch.nn.Sequential(*features) 
        self.model = self.model.to(self.device) 

        self.data_dir = './split_dataset/'
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),data_transforms[x])for x in ['train', 'val', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=32,shuffle=True, num_workers=16)for x in ['train', 'val', 'test']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        
        self.train_Loss_list = []
        self.train_Accuracy_list = []
        self.val_Loss_list = []
        self.val_Accuracy_list = []
    def data_transform(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor()
            ])
        } 
        return data_transforms
    def imshow(self, inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    def train_model(self, criterion, optimizer, scheduler, num_epochs=25):
        print(self.device)
        since = time.time()
     # Create a temporary directory to save training checkpoints
        best_model_params_path = 'vgg19_best.pt'
        torch.save(self.model.state_dict(), best_model_params_path)
        best_acc = 0.0
        self.num_epochs = num_epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                self.load_loss_accuracy(epoch_loss, epoch_acc, phase)
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model, best_model_params_path)

            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model = (torch.load(best_model_params_path))
        return self.model
    def visualize_model(self, num_images=6):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.class_names[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)
    def load_loss_accuracy(self, loss, acc, phase):
        if phase == 'train':
            if isinstance(loss, torch.Tensor):
                loss = loss.cpu().numpy()
            self.train_Loss_list.append(loss)  # Convert tensor to numpy array
            self.train_Accuracy_list.append(acc)
        elif phase == 'val':
            if isinstance(loss, torch.Tensor):
                loss = loss.cpu().numpy()            
            self.val_Loss_list.append(loss)   # Convert tensor to numpy array
            self.val_Accuracy_list.append(acc)
    def plot_train_loss_acc(self):
        x1 = range(0, self.num_epochs)
        x2 = range(0, self.num_epochs)
        y1 = [item.cpu().numpy() for item in self.train_Accuracy_list]
        y2 = self.train_Loss_list
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, '.-')
        plt.title('Train accuracy vs. epoches')
        plt.ylabel('Train accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Train loss vs. epoches')
        plt.ylabel('Train loss')
        plt.savefig("vgg19_Train_accuracy_loss.jpg")
        plt.show()
    def plot_val_loss_acc(self):
        x1 = range(0, self.num_epochs)
        x2 = range(0, self.num_epochs)
        y1 = [item.cpu().numpy() for item in self.val_Accuracy_list]
        y2 = self.val_Loss_list
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, '.-')
        plt.title('Val accuracy vs. epoches')
        plt.ylabel('Val accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Val loss vs. epoches')
        plt.ylabel('Val loss')
        plt.savefig("vgg19_Val_accuracy_loss.jpg")
        plt.show()
    def plot_confusion_matrix(self):
        y_pred = []
        y_true = []
        plt.figure(figsize=(20, 20))

        self.model.eval()
        for inputs, labels in self.dataloaders['test']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            output = self.model(inputs)
            _, preds = torch.max(output, 1)                            # preds是預測結果
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_true.extend(labels.view(-1).detach().cpu().numpy())      # target是ground-truth的label
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, self.class_names, self.class_names)     # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Image-Classification-using-PyTorch.html
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig("vgg19_confusion_matrix.png")
if __name__ == "__main__":
    resnet50_model = models.vgg19(pretrained=True)
    num_class = 45
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(resnet50_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    train = train(resnet50_model, num_class)
    train.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 30)
    train.visualize_model(num_images=10)
    train.plot_train_loss_acc()
    train.plot_val_loss_acc()
    train.plot_confusion_matrix()