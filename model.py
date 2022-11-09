# system import
import argparse
from tqdm import tqdm
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# helper packages
import pandas as pd

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

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss

def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

def save_checkpoint(state, is_best,
                    file_folder="./outputs/",
                    filename='checkpoint.pth.tar'):
    """save checkpoint"""
    if not os.path.exists(file_folder):
        os.makedirs(os.path.expanduser(file_folder), exist_ok=True)
    torch.save(state, os.path.join(file_folder, filename))
    if is_best:
        # skip the optimization state
        state.pop('optimizer', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))

def main(args):
    model = ConvNeuralNet()

    training_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_df = pd.read_csv("./sign_lang_mnist/sign_mnist_test.csv")
    test_df = pd.read_csv("./sign_lang_mnist/sign_mnist_test.csv")

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = torch.utils.data.DataLoader(train_df, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_df, batch_size=args.batch_size, shuffle=False)

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        # train model for 1 epoch
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        # evaluate the model on test_set after this epoch
        acc = test_model(model, test_loader, epoch)
        # save the current checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc' : max(best_acc, acc),
            'optimizer' : optimizer.state_dict(),
            }, (acc > best_acc))
        best_acc = max(best_acc, acc)
    
    torch.save(model.state_dict(), "./model.pt")
    print("Finished Training")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification using Pytorch')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='number of images within a mini-batch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    main(args)

