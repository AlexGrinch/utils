from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np


NUM_CLASSES = 8


class CorpusDataset(Dataset):
    def __init__(self, data_path):
        t = np.load(data_path)
        self.data, self.labels = t["data"], t["labels"]
        self.data = self.data.astype(np.float32)[:,None,:,:]
        self.labels = self.labels.astype(np.int)
        if data_path[-5] == "n" or data_path[-6] == "n":
            samples_per_class = self.data.shape[0] // 8
            self.data = self.data[:NUM_CLASSES*samples_per_class]
            self.labels = self.labels[:NUM_CLASSES*samples_per_class]
        else:
            self.data = self.data[:NUM_CLASSES*200]
            self.labels = self.labels[:NUM_CLASSES*200]
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        img, lbl = self.data[index], self.labels[index]
        dct = {"images": img, "labels": lbl}
        return dct

    def __len__(self):
        return self.len


class CorpusSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.indices = np.arange(self.len)
        np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.len


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.drop(x)
        x = F.max_pool2d(x, 2, 2)
        return x
    
class Net(nn.Module):
    def __init__(self, dropout=0.2):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1, 16, 5, 1, dropout)
        self.conv2 = ConvBlock(16, 16, 4, 1, dropout)
        self.conv3 = ConvBlock(16, 16, 3, 1, dropout)
        self.conv4 = ConvBlock(16, 16, 3, 1, dropout)
        self.fc1 = nn.Linear(26*26*16, 512)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 26*26*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch["images"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data, target = batch["images"].to(device), batch["labels"].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = DataLoader(
        dataset=CorpusDataset("data/convex_clean_train.npz"),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(
        dataset=CorpusDataset("data/convex_clean_test.npz"),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
    print (f"BEST ACC: {best_acc}")

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()