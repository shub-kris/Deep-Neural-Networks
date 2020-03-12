import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


batch_size = 100
nz = 100
out_dim = 28 * 28
lr = 0.0002
num_epochs = 200

transform = transforms.Compose([
    transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./datasets/MNIST', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.fcG = nn.Sequential(
			nn.Linear(nz, 256),
			nn.LeakyReLU(0.2, inplace = True),
			
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2, inplace = True),
			
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace = True),
			
			nn.Linear(1024, out_dim),
			nn.Tanh()
		)
		
	def forward(self, x):
		x = self.fcG(x)
		return x

netG = Generator().to(device)
print(netG)		
		
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		
		self.fcD = nn.Sequential(
			nn.Linear(out_dim, 1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3),
			
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3),
			
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3),
			
			nn.Linear(256,1),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		x = self.fcD(x)
		return x


netD = Discriminator().to(device)
print(netD)

criterion = nn.BCELoss()

noise = torch.randn(batch_size,nz, device = device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr = lr)
optimizerG = optim.Adam(netG.parameters(), lr = lr)

G_losses = []
D_losses = []

for epoch in range(num_epochs):
    for i, (data,_) in enumerate(dataloader):
        
        netD.zero_grad()
        
        real_x = data.view(-1, out_dim).to(device)
        b_size = real_x.size(0)
        label = torch.full((b_size,), real_label, device = device)
        
        output = netD(real_x).view(-1)
        
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(b_size, nz, device = device)
        fake_x = netG(noise)
        label.fill_(fake_label)
        
        output = netD(fake_x.detach()).view(-1)
        
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        
        optimizerD.step()
        
        netG.zero_grad()
        label.fill_(real_label)
        
        output = netD(fake_x).view(-1)
        
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        optimizerG.step()
        
    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                 % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        
with torch.no_grad():
    test_z = Variable(torch.randn(batch_size, nz).to(device))
    generated = netG(test_z)
    save_image(generated.view(generated.size(0), 1, 28, 28), './sample_' + '.png')	
	
	







			