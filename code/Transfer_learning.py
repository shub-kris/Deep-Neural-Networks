import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import copy

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        _, pred = torch.max(output, 1)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    

def eval(model, val_loader, criterion, device):
    model.eval()
    eval_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
        
            data = data.to(device)
            target = target.to(device)
        
            output = model(data)
            _, pred = torch.max(output, 1)
            
            eval_loss += criterion(output, target).item()
            correct += torch.sum(pred == target.data)
            
    eval_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        eval_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    acc = 100. * correct / len(val_loader.dataset)
    
    return (acc.item(), model.state_dict())

def main():

    batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    lr = 0.003

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_data_path = './datasets/hymenoptera_data/train'
    val_data_path = './datasets/hymenoptera_data/val'

    train_dataset = datasets.ImageFolder(root = train_data_path, transform = train_transform)
    val_dataset = datasets.ImageFolder(root = val_data_path, transform = val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    out_class = 2 #bees, ants
    
    switch = 1    # 1 if you want to transfer learning using finetuning the last layer weights only
    
    Resnet18 = models.resnet18(pretrained = True)
    
    if(switch):
        for param in Resnet18.parameters():
            param.requires_grad = False
    
    num_features = Resnet18.fc.in_features
    
    Resnet18.fc = nn.Linear(num_features, out_class)
    
    Resnet18 = Resnet18.to(device)
    

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(Resnet18.parameters(), lr=0.001, momentum=0.9)
    
    best_model_wt = copy.deepcopy(Resnet18.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        train(Resnet18, train_loader, criterion, optimizer, epoch, device)
        epoch_acc, epoch_model_wt = eval(Resnet18, val_loader, criterion, device)
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_model_wt = copy.deepcopy(epoch_model_wt)
    Resnet18.load_state_dict(best_model_wt)
    print(best_acc)
if __name__ == '__main__':
    main()

