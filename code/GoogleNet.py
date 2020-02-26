import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import argparse


class Inception(nn.Module):
    def __init__(self,in_ch,out_ch1,mid_ch13,out_ch13,mid_ch15,out_ch15,out_pool_conv):
        super(Inception,self).__init__()
        
        # 1*1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch1,kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch1)
            )
        
        # 1*1 -> 3*3
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch13,kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_ch13),
            nn.Conv2d(mid_ch13,out_ch13,kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch13)
            )
            
            
        # 1*1 -> 5*5
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch15,kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_ch15),
            nn.Conv2d(mid_ch15,out_ch15,kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch15)
            )
        # 3*3 pool -> 1*1 conv1
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_ch,out_pool_conv,kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_pool_conv)
            )
            
    def forward(self,x):
        y1 = self.conv1(x)
        y2 = self.conv13(x)
        y3 = self.conv15(x)
        y4 = self.pool_conv(x)
        return torch.cat([y1,y2,y3,y4],1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        
        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.Inception1 = Inception(32,64,64,64,32,32,64)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Inception2 = Inception(224,128,128,128,48,48,96)
        self.Inception3 = Inception(400,256,256,256,16,16,32)
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(560 * 4 *4,1024)
        self.fc2 = nn.Linear(1024,10)
            

    def forward(self,x):
        x = self.base_layer(x)
        x = self.Inception1(x)
        x = self.maxpool(x)
        x = self.Inception2(x)
        x = self.maxpool(x)
        x = self.Inception3(x)
        x = self.avgpool(x)
        x = x.view(-1, 4*4*560)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch,criterion):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred, pred_class = torch.max(output, dim = 1)
            test_loss += criterion(output, target).item()
            correct += pred_class.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        
        
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])  
    
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./datasets/CIFAR10', train=True, download=True,
                       transform= transform_train),
        batch_size=args.batch_size, shuffle=True)
        
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./datasets/CIFAR10', train=False, transform= transform_test),
        batch_size=args.test_batch_size, shuffle=False)

    model = GoogleNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 20):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(args, model, device, test_loader, criterion)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "GoogleNet.pt")


if __name__ == '__main__':
    main()