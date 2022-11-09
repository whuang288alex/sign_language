import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.csvfile = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
file = pd.read_csv("./sign_lang_mnist/sign_mnist_test.csv")
print(file)
