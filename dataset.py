import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset

class ASLDataset(Dataset):
    def __init__(self, file_path, train=True):
        self.dataset = pd.read_csv(file_path)
        self.labels = self.dataset['label']
        self.train = train
        self.img_size = 224

        # concatnate pixels
        self.imgs = torch.zeros((self.dataset.shape[0], 1))
        for i in range(1,785):
            pixel = 'pixel' + str(i)
            current_column = self.dataset[pixel]
            current_column = torch.FloatTensor(current_column).unsqueeze(1)
            self.imgs = torch.cat((self.imgs,current_column), 1)

        self.imgs = self.imgs[:, 1::]
        self.imgs = self.imgs.view(-1,28,28)

    def __getitem__(self, i):
        img = self.imgs[i]
        img = img.numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        img /= 255.

        if self.train:
            return img, self.labels[i]
        return img

    def __len__(self):
        return self.imgs.shape[0]

'''
df = CustomDataset("./sign_lang_mnist/sign_mnist_test.csv")
labels = df.labels
print(df.labels)
'''