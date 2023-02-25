import numpy as np
import gzip
from torch.utils.data import Dataset


def load_data(image_path, label_path):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    """
    with gzip.open(label_path, 'rb') as lbPath:  # rb表示的是读取二进制数据
        label = np.frombuffer(lbPath.read(), np.uint8, offset=8)
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    """
    with gzip.open(image_path, 'rb') as imgPath:
        image = np.frombuffer(imgPath.read(), np.uint8, offset=16).reshape(len(label), 28, 28)
    return image, label


class CustomImageDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, target_transform=None):
        self.image, self.label = load_data(image_path, label_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

