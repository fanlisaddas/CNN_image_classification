import torch
from torch import nn
from data_load import CustomImageDataset
from model import Model
from train_and_test import train_loop, test_loop
from torch.utils.data import DataLoader
from torchvision import transforms


# 训练集文件
training_image_path = 'data/train-images-idx3-ubyte.gz'
# 训练集标签文件
training_label_path = 'data/train-labels-idx1-ubyte.gz'

# 测试集文件
test_images_path = 'data/t10k-images-idx3-ubyte.gz'
# 测试集标签文件
test_label_path = 'data/t10k-labels-idx1-ubyte.gz'


training_data = CustomImageDataset(training_image_path, training_label_path, transform=transforms.ToTensor())

test_data = CustomImageDataset(test_images_path, test_label_path, transform=transforms.ToTensor())

learning_rate = 1e-3
batch_size = 128
epochs = 20
SEED = 0  # 时间种子

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 设置时间种子
torch.manual_seed(SEED)
# 检查设备
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    device = "cuda"
else:
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

print(f"Using {device} device")

model = Model().to(device)
print(model)

# 定义损失函数和优化策略
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
