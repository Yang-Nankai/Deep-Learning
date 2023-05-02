# %% [markdown]
# ### MLP-Mixer进行手写体识别
# 对于MNIST手写体识别任务，可以使用MLP-Mixer模型来构建一个分类器。具体来说，可以将MNIST图像转换为向量形式，并将其输入到MLP-Mixer模型中进行分类。在训练过程中，可以使用交叉熵损失函数来衡量模型在MNIST数据集上的分类性能，并使用优化算法对模型参数进行更新。在测试过程中，可以使用训练好的模型对新的手写数字图像进行分类。
# [参考学习](https://arxiv.org/pdf/2105.01601.pdf)

# %%
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from einops.layers.torch import Rearrange, Reduce
from functools import partial
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
%matplotlib inline


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

# %% [markdown]
# ## Data
#
# Next we'll load the MNIST data.  First time we may have to download the data, which can take a while.
#
# Note that we are here using the MNIST test data for *validation*, instead of for testing the final model.

# %%
batch_size = 32  # 批处理大小

# 下载数据集
train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())
# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

# %% [markdown]
# MLP-Mixer的实现
# 这里参考的[mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch)
#

# %%


def pair(x): return x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w %
                                            patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                  p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(
                num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(
                dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

# %% [markdown]
# 定义相关超参数并使用超参数初始化模型


# %%

n_epochs = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
img_height = 28
img_width = 28

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MLPMixer(
    image_size=28,
    patch_size=7,
    dim=14,
    depth=3,
    num_classes=10,
    channels=1
)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()


# %% [markdown]
# 定义训练函数

# %%
# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size_train = data.shape[0]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pre_out = model(data)
        targ_out = torch.nn.functional.one_hot(target, num_classes=10)
        targ_out = targ_out.view((batch_size_train, 10)).float()
        loss = mse(pre_out, targ_out)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# %% [markdown]
# 定义测试函数

# %%
# 定义测试函数


def validate(model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size_test = data.shape[0]
            data, target = data.to(device), target.to(device)
            pre_out = model(data)
            targ_out = torch.nn.functional.one_hot(target, num_classes=10)
            targ_out = targ_out.view((batch_size_test, 10)).float()
            test_loss += mse(pre_out, targ_out)  # 将一批的损失相加
            # get the index of the max log-probability
            pred = pre_out.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct.to(torch.float32) / \
        len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset), accuracy))

# %% [markdown]
# 进行测试迭代，并保存相关模型


# %%
for epoch in range(n_epochs):
    train(model, DEVICE, train_loader, optimizer, epoch)
    validate(model, DEVICE, validation_loader)
    torch.save(model.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')
