###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def conv_block(in_channel, out_channel, relu_last=True, **kwargs):
    layers = [nn.Conv2d(in_channel, out_channel, bias=False, **kwargs),
              nn.BatchNorm2d(out_channel)]
    if relu_last:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
class ResidualSEBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride, r=16):
        super(ResidualSEBlock, self).__init__()
        self.residual = nn.Sequential(
            conv_block(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            conv_block(out_channel, out_channel * self.expansion, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.shortcut = conv_block(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride,
                                       relu_last=False)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channel * self.expansion, out_channel * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel * self.expansion // r, out_channel * self.expansion),
            nn.Sigmoid())

    def forward(self, x):
        r = self.residual(x)
        bs, c, _, _ = r.shape
        s = self.squeeze(r).view(bs, c)
        e = self.excitation(s).view(bs, c, 1, 1)
        return F.relu(self.shortcut(x) + r * e.expand_as(r))
class SEResnet(nn.Module):
    def __init__(self, in_channel, n_classes, num_blocks, block):
        super(SEResnet, self).__init__()
        self.in_channels = 64
        self.feature = nn.Sequential(
            conv_block(in_channel, 64, kernel_size=3, padding=1),
            self._make_stage(64, 1, num_blocks[0], block),
            self._make_stage(128, 2, num_blocks[1], block),
            self._make_stage(256, 2, num_blocks[2], block),
            self._make_stage(512, 2, num_blocks[3], block)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_channels, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def _make_stage(self, out_channel, stride, num_block, block):
        layers = []
        for i in range(num_block):
            stride = stride if i == 0 else 1
            layers.append(block(self.in_channels, out_channel, stride))
            self.in_channels = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.feature(x))
def seresnet18():
    return SEResnet(3, 10, [2, 2, 2, 2], ResidualSEBlock)

def seresnet34():
    return SEResnet(3, 10, [3, 4, 6, 3], ResidualSEBlock)
net = seresnet18().to(device)
print(net)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def train(epoch, log_interval=2000):
    # Set model to training mode
    net.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(trainloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = net(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()  
        
        # Update weights
        optimizer.step()    #  w - alpha * dL / dw
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item()))
def validate(loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct = 0, 0
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(testloader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(testloader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(testloader.dataset), accuracy))
# validate
# epochs = 10

# lossv, accv = [], []
# for epoch in range(1, epochs + 1):
#     train(epoch)
#     validate(lossv, accv)
    
# print("lossv: ", lossv)
# print("accv: ", accv)

for epoch in range(10):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(labels)
        # put data into GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
PATH = './results/cifar_seresnet.pth'
torch.save(net.state_dict(), PATH)
print("save success!")