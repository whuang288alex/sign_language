import torch
import torch.nn as nn

class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        
        # reusable layers
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # convolution layer #1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=75, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(75)
        
        # convolution layer #2
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=50, kernel_size=3)
        self.dropout1 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm2d(50)
        
        # convolution layer #1
        self.conv3 = nn.Conv2d(in_channels = 50, out_channels=25, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(25)
        
        # fully connected layers
        self.fc1 = nn.Linear(25, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 24)
        
            
    # Progresses data across layers    
    def forward(self, x):
        
        # convolution layer #1
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.max_pool(out)
        
        # convolution layer #2
        out = self.relu(self.conv2(out))
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.max_pool(out)
        
        # convolution layer #3
        out = self.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.max_pool(out)
        
        # fully connected layers
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))
        return out