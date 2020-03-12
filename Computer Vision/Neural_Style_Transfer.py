import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512
lr = 0.01
num_epochs = 600
#We need to convert the image into Tensor
transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def load_image(img_name):
    img = Image.open(img_name)
    img = transform(img).unsqueeze(0)   #Adds an extra dimension to match i/p dimension
    return img.to(device, torch.float)
    
style_img = load_image('./picasso.jpg')
content_img = load_image('./dancing.jpg')

def plot_img(tensor, title):
    img = tensor.cpu().clone().detach()
    img = img.numpy().squeeze(0)
    img = img.transpose(1,2,0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array(
    (0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    # img = transforms.ToPILImage(img)
    plt.imshow(img)
    plt.title(title)
    plt.show()
    plt.pause(0.001)
    
    
plt.figure()
plot_img(style_img, title = 'Style Image')

plt.figure()
plot_img(content_img, title = ' Content Image')


VGG19 = models.vgg19(pretrained = True)
# print(VGG19)

#We don't need to retrain VGG19
for param in VGG19.parameters():
    param.requires_grad_(False)

def extract_features(img, model, layers = None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',    # Content Layer
            '28': 'conv5_1'
            }
            
    features = {}
    x = img
    for name, layer in enumerate(model.features):     # It extracts output from the layers and checks the one that we want
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

#Now we need to define the Gram_Matrix for the style loss

def gram_matrix(tensor):
    _, nc, h, w = tensor.size()
    tensor = tensor.view(nc, h * w)
    gram = torch.mm(tensor, tensor.t())  # Computing Correlation Matrix
    return gram 


#Now replace maxpool to AvgPool
for i, layer in enumerate(VGG19.features):
    if isinstance(layer, torch.nn.MaxPool2d):     #If object is of the class (if layer is of class Maxpool)
        VGG19.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

VGG19.to(device).eval()

content_features = extract_features(content_img, VGG19)
style_features = extract_features(style_img, VGG19)

#Compute Gram Matrix for each layer in style_features

style_gram_layer = {layer: gram_matrix(style_features[layer]) for layer in style_features}

#Let's create a random image for the transformation

generated_img = torch.randn_like(content_img).requires_grad_(True).to(device)  #Here our parameters are PIXELS of the image

#Now we define weight for different layers of the Gram Matrix

style_weights_loss = {
                      'conv1_1': 0.75,
                      'conv2_1': 0.5,
                      'conv3_1': 0.2,
                      'conv4_1': 0.2,
                      'conv5_1': 0.2}
                      
#Total loss is weighted linear combination of Style Loss and Content Loss

content_wt = 1e4
style_wt = 1e2

optimizer = optim.Adam([generated_img], lr = lr)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    img_features = extract_features(generated_img, VGG19)
    
    content_loss = torch.mean((img_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    style_loss = 0
    
    for layer in style_weights_loss:
        img_feature = img_features[layer]
        img_gram = gram_matrix(img_feature)
        _, nc, h, w = img_feature.shape
        style_gram = style_gram_layer[layer]    #Gram matrix for the styled image for this layer
        layer_style_loss = style_weights_loss[layer] * torch.mean((img_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (nc* h * w)
        
        total_loss = content_wt * content_loss + style_wt * style_loss
        total_loss.backward(retain_graph = True)
        optimizer.step()
    
    if epoch % 20 == 0:
        total_loss_rounded = round(total_loss.item(), 2)
        content_fraction = round(
        content_wt*content_loss.item()/total_loss.item(), 2)
        style_fraction = round(
        style_wt*style_loss.item()/total_loss.item(), 2)
        print('Iteration {}, Total loss: {} - (content: {}, style {})'.format(
            epoch,total_loss_rounded, content_fraction, style_fraction))

plot_img(generated_img,'Neural Image')
            
        