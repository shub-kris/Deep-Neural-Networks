import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import argparse


class ResBlock(nn.Module):

    def __init__(self,in_ch,out_ch, stride = 1):
        super(ResBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_ch)
        )
        
        self.add = nn.Sequential()
        if stride != 1 or in_ch != out_ch :
            self.add = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = stride),
            nn.BatchNorm2d(out_ch)
            )
        
    def forward(self,x):
        out = self.conv(x)
        out += self.add(x)
        return out
    

class ResNet(nn.Module):

        def __init__(self, resblock, num_blocks, num_classes = 10):
            super(ResNet,self).__init__()
            self.in_ch = 64
            
            self.conv1 = nn.Conv2d(3,64,kernel_size = 1, stride = 1, padding = 1)
            self.bn1 = nn.BatchNorm2d(64)
            
            self.layer1 = self._make_layer(resblock, 64, num_blocks[0], stride = 1)
            self.layer2 = self._make_layer(resblock, 128, num_blocks[1], stride = 2)
            self.layer3 = self._make_layer(resblock, 256, num_blocks[2], stride = 2)
            # self.layer4 = self._make_layer(resblock, 512, num_blocks[3], stride = 2)
            self.linear = nn.Linear(256, num_classes)
            
            
        def _make_layer(self, resblock, out_ch, num_blocks, stride):
            layers = []
            for i in range(num_blocks):
                layers.append(resblock(self.in_ch, out_ch, stride))
                self.in_ch = out_ch
                
            return nn.Sequential(*layers)
            
        def forward(self,x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            # out = self.layer4(out)
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
            return out
    
    
def ResNet18():
    return ResNet(ResBlock,[2,2,2,2])

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
                        help='number of epochs to train (default: 14)')
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

    model = ResNet18().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 20):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(args, model, device, test_loader, criterion)

    if args.save_model:
        torch.save(model.state_dict(), "ResNet.pt")


if __name__ == '__main__':
    main()