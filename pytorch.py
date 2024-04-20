import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm


def create_dataframe(path, label_dict):
    df = pd.DataFrame({"path": [], "label": [], "class_id": []})
    folder_list = os.listdir(path)
    for folder in folder_list:
        img_path = os.path.join(path, folder)
        jpg_list = glob(img_path + '/*.jpg')
        for jpg in jpg_list:
            new_data = pd.DataFrame({"path": jpg, "label": folder, "class_id": label_dict[folder]}, index=[1])
            df = pd.concat([df, new_data], ignore_index=True)
    df[["path"]] = df[["path"]].astype(str)
    df[["label"]] = df[["label"]].astype(str)
    df[["class_id"]] = df[["class_id"]].astype(int)
    return df


train_path = r'data\train'
val_path = r'data\val'
label_dict = {"healthy": 0, "angular_leaf_spot": 1, "bean_rust": 2}

train_df = create_dataframe(train_path, label_dict)
val_df = create_dataframe(val_path, label_dict)

train_transforms = v2.Compose([
    v2.Resize(256),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.RandomErasing(p=0.5, scale=(0.1, 0.15)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms_):
        self.df = dataframe
        self.transforms_ = transforms_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.iloc[index]['path']
        img = Image.open(image_path).convert("RGB")
        transformed_img = self.transforms_(img)
        class_id = self.df.iloc[index]['class_id']
        return transformed_img, class_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 2 if device == 'cuda' else 4
train_dataset = MyDataset(train_df, train_transforms)
val_dataset = MyDataset(val_df, test_transforms)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class_size = 3
model = models.efficientnet_v2_s(weights='DEFAULT')
model.classifier[1] = torch.nn.Linear(1280, class_size)


def train(dataloader, model, loss_fn, optimizer, lr_scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    for data_, target_ in dataloader:
        target_ = target_.type(torch.LongTensor)
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        outputs = model(data_)
        loss = loss_fn(outputs, target_)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        epoch_correct += torch.sum(pred == target_).item()
    lr_scheduler.step()
    return epoch_correct / size, epoch_loss / num_batches


class LongTensor:
    pass


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_loss = 0.0
    epoch_correct = 0
    with torch.no_grad():
        model.eval()
        for data_, target_ in dataloader:
            target_ = target_.type(torch.LongTensor)
            data_, target_ = data_.to(device), target_.to(device)
            outputs = model(data_)
            loss = loss_fn(outputs, target_)
            epoch_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            epoch_correct += torch.sum(pred == target_).item()
    return epoch_correct / size, epoch_loss / num_batches


model.to(device)
EPOCHS = 50
logs = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_milestones = [7, 14, 21, 28, 35]
multi_step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
patience = 5
counter = 0
best_loss = np.inf

for epoch in tqdm(range(EPOCHS)):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, multi_step_lr_scheduler)
    val_acc, val_loss = test(val_loader, model, criterion)
    print(f'EPOCH: {epoch} \
    train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} \
    val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f} \
    Learning Rate: {optimizer.param_groups[0]["lr"]}')
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)
    logs['val_loss'].append(val_loss)
    logs['val_acc'].append(val_acc)
    torch.save(model.state_dict(), "last.pth")
    if val_loss < best_loss:
        counter = 0
        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
            torch.save(model.state_dict(), "best.pth")
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs['train_loss'], label='Train_Loss')
        plt.plot(logs['val_loss'], label='Validation_Loss')
        plt.title('Train_Loss & Validation_Loss', fontsize=20)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(logs['train_acc'], label='Train_Accuracy')
        plt.plot(logs['val_acc'], label='Validation_Accuracy')
        plt.title('Train_Accuracy & Validation_Accuracy', fontsize=20)
        plt.legend()

        model.load_state_dict(torch.load('best.pth'))
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for data_, target_ in tqdm(val_loader):
                target_ = target_.type(torch.LongTensor)
                data_, target_ = data_.to(device), target_.to(device)
                outputs = model(data_)
                _, pred = torch.max(outputs, dim=1)
                y_true.extend(target_.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        ax = plt.subplot()
        CM = confusion_matrix(y_true, y_pred)
        sns.heatmap(CM, annot=True, fmt='g', ax=ax, cbar=False, cmap='RdBu')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        plt.show()

        target_names = ["healthy", "angular_leaf_spot", "bean_rust"]
        clf_report = classification_report(y_true, y_pred, target_names=target_names)
        print(clf_report)

        Acc = accuracy_score(y_true, y_pred)
        print("Accuracy is: {0:.3f}%".format(Acc * 100))
