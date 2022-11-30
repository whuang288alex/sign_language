# system import
from tqdm import tqdm
import os

# torch import
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.io import read_image

# data and model import
from dataset import ASLDataset
from model import ConvNeuralNet

device = "cuda"
epochs = 20
learning_rate = 0.001
batch_size = 128
train_data_path = "../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv"
test_data_path = "../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv"

def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output = model(input.to(device))
        loss = criterion(output.to(device), target.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss

def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input.to(device))
            # softmax to get probability
            predicted = torch.softmax(output,dim=1) 
            # get predicted labels of test data
            _, predicted = torch.max(predicted, 1) 
            # remove cuda label
            predicted = predicted.data.cpu() 
            
            correct += target.eq(predicted).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(epoch+1, 100. * test_acc))

    return test_acc

def main():
    train_df = ASLDataset(train_data_path)
    test_df = ASLDataset(test_data_path)

    train_loader = torch.utils.data.DataLoader(train_df, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_df, batch_size=64, shuffle=False)

    model = ConvNeuralNet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose= True, min_lr=1e-6)
    
    best_acc = 0.0
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        # train model for 1 epoch
        train_model(model, train_loader, optimizer, criterion, epoch)
        # evaluate the model on test_set after this epoch
        acc = test_model(model, test_loader, epoch)
        best_acc = max(best_acc, acc)
    
    torch.save(model.state_dict(), "./model.pt")
    print("Finished Training")

main()