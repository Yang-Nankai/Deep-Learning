# %% [markdown]
# # MNIST handwritten digits classification with MLPs
#
# In this notebook, we'll train a multi-layer perceptron model to classify MNIST digits using **PyTorch**.
#
# First, the needed imports.

# %%
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from itertools import cycle
import random
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
# 设置随机种子防止保证每次随机的样本都是一致的

# %%
# 设置随机种子
# 固定shuffle随机数种子以及cpu等backend算法
seed = 10
random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的   　　
torch.backends.cudnn.deterministic = True

# %% [markdown]
# The train and test data are provided via data loaders that provide iterators over the datasets. The first element of training data (`X_train`) is a 4th-order tensor of size (`batch_size`, 1, 28, 28), i.e. it consists of a batch of images of size 1x28x28 pixels. `y_train` is a vector containing the correct classes ("0", "1", ..., "9") for each training digit.

# %%
# 读取loader中的张量大小
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

# %% [markdown]
# Here are the first 10 training digits:

# %%
# 显示MNIST图片
pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap="gray_r")
    plt.title('Class: '+str(y_train[i].item()))

# %% [markdown]
# ## MLP network definition
#
# Let's define the network as a Python class.  We have to write the `__init__()` and `forward()` methods, and PyTorch will automatically generate a `backward()` method for computing the gradients for the backward pass.
#
# Finally, we define an optimizer to update the model parameters based on the computed gradients.  We select *stochastic gradient descent (with momentum)* as the optimization algorithm, and set *learning rate* to 0.01.  Note that there are [several different options](http://pytorch.org/docs/optim.html#algorithms) for the optimizer in PyTorch that we could use instead of *SGD*.

# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)  # weight: [28*28, 50]   bias: [50, ]
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 80)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, 10)

#         self.relu1 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)   # [32, 28*28]
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)   # [32, 10]
        return F.log_softmax(self.fc3(x), dim=1)


model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)

# %% [markdown]
# ## Learning
#
# Let's now define functions to `train()` and `validate()` the model.

# %%


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()  # w - alpha * dL / dw

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


# %%


def validate(loss_vector, accuracy_vector):
    # 补充ROC曲线
    model.eval()
    val_loss, correct = 0, 0

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        score_temp = output  # (batchsize, nclass)
        score_list.extend(score_temp.detach().cpu().numpy())
        label_list.extend(target.cpu().numpy())
    '''
    num_class = 10
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.savefig('set113_roc.jpg')
    plt.show()
    '''

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / \
        len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


# %% [markdown]
# Now we are ready to train our model using the `train()` function.  An *epoch* means one pass through the whole training data. After each epoch, we evaluate the model using `validate()`.

# %%
% % time
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)

# %% [markdown]
# Let's now visualize how the training progressed.
#
# * *Loss* is a function of the difference of the network output and the target values.  We are minimizing the loss function during training so it should decrease over time.
# * *Accuracy* is the classification accuracy for the test data.

# %%
plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs+1), lossv)
plt.title('validation loss')

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs+1), accv)
plt.title('validation accuracy')

# %% [markdown]
# ## Model tuning
#
# Modify the MLP model.  Try to improve the classification accuracy, or experiment with the effects of different parameters.  If you are interested in the state-of-the-art performance on permutation invariant MNIST, see e.g. this [recent paper](https://arxiv.org/abs/1507.02672) by Aalto University / The Curious AI Company researchers.
#
# You can also consult the PyTorch documentation at http://pytorch.org/.

# %% [markdown]
# 查看迭代次数对于Accuracy的影响

# %%
% % time
# 查看迭代次数对于Accuracy的影响
plt.figure(figsize=(5, 3))
for style, width, epochs in (('g-', 1, 1), ('b--', 1, 5), ('r-+', 1, 10)):
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)
    plt.plot(np.arange(1, epochs+1), accv, style,
             label='epochs '+str(epochs), linewidth=width)

plt.title('Epochs-validation accuracy')
plt.axis([0, 20, 97, 99])
plt.legend()
plt.show()

# %% [markdown]
# 查看学习率对于Accuracy的影响

# %%
% % time
# 查看迭代次数对于Accuracy的影响
epochs = 5

plt.figure(figsize=(5, 3))
for style, width, lr in (('g-', 1, 0.01), ('b--', 2, 0.001), ('r-+', 1, 0.1), ('p-', 2, 0.5)):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)
    plt.plot(np.arange(1, epochs+1), lossv, style,
             label='lr '+str(lr), linewidth=width)

plt.title('LearningRate-validation loss')
plt.legend()
plt.show()

# %% [markdown]
# 选择不同的optim优化策略

# %%
% % time
# 修改尝试不同的优化器
epochs = 3

plt.figure(figsize=(5, 3))
for style, width, title, optimizer in (('g-', 1, 'SGD', torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)), ('b--', 2, 'RMSprop', torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.5)), ('r-+', 1, 'Adagrad', torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)), ('p-', 2, 'Adam', torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99)))):
    optimizer = optimizer
    criterion = nn.CrossEntropyLoss()
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)
    plt.plot(np.arange(1, epochs+1), accv, style,
             label='optimizer '+str(title), linewidth=width)

plt.title('optimizer-validation accv')
plt.legend()
plt.show()

# %% [markdown]
# 增加一层隐藏层看是否对结果有优化

# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)  # weight: [28*28, 50]   bias: [50, ]
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 80)
        self.fc2_drop = nn.Dropout(0.2)
        # 增加一层
        self.fc3 = nn.Linear(80, 60)
        self.fc3_drop = nn.Dropout(0.2)
        self.fc4 = nn.Linear(60, 10)

#         self.relu1 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)   # [32, 28*28]
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)   # [32, 10]
        # 增加一层
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)   # [32, 10]
        return F.log_softmax(self.fc4(x), dim=1)


model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)

# %%
% % time
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)

# %%
plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs+1), lossv)
plt.title('validation loss')

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs+1), accv)
plt.title('validation accuracy')
