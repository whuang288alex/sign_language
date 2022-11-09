import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=75, kernel_size=3)
        
        self.bn1 = nn.BatchNorm2d(75)
        
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=50, kernel_size=3)
        
        self.dropout1 = nn.Dropout(0.2)
        
        self.bn2 = nn.BatchNorm2d(50)
        
        self.conv3 = nn.Conv2d(in_channels = 50, out_channels=25, kernel_size=3)
        
        self.bn3 = nn.BatchNorm2d(25)
        
        self.fc1 = nn.Linear(25, 512)
        
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 24)
        
            
    # Progresses data across layers    
    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.max_pool(out)
        
        out = self.relu(self.conv2(out))
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.max_pool(out)
        
        out = self.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.max_pool(out)
        
     
        out = self.flatten(out)
        

        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))
        return out


