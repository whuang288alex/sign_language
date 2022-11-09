import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    
    def __init__(self, file_path, transform=None, target_transform=None):
        self.dataset = pd.read_csv(file_path)
        self.labels = self.dataset['label']
        self.imgs = self.dataset.iloc[:, 1::]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs.iloc[idx]
        label = self.labels.iloc[idx]
        
        print("idx:{} label:{}".format(idx, label-1))
        array = np.array(image, dtype=np.uint8).reshape(28,28)
        image = Image.fromarray(array)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

'''
df = CustomDataset("./sign_lang_mnist/sign_mnist_test.csv")
img, label = df[0]
print(img, "\nlabel:", label)
'''
