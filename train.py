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


############################ DATA LOADERS ############################
class CorpusDataset(Dataset):
    def __init__(self, data_path, augment=False):
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
        if augment:
            self.data = np.vstack((
                self.data, self.data[:,:,:,::-1],
                self.data[:,:,::-1,:], self.data[:,:,::-1,::-1]))
            self.labels = np.concatenate((
                self.labels, self.labels,
                self.labels, self.labels))
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        img, lbl = self.data[index], self.labels[index]
        #img = np.vstack((img, 1-img))
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

############################ NETWORKS ############################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = F.max_pool2d(x, 2, 2)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1, 8, 5, 1)
        self.conv2 = ConvBlock(8, 16, 4, 1)
        self.conv3 = ConvBlock(16, 32, 3, 1)
        self.conv4 = ConvBlock(32, 64, 3, 1)
        self.fc1 = nn.Linear(26*26*64, 512)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 26*26*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

############################ TRAINING ############################
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
    errors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data, target = batch["images"].to(device), batch["labels"].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            
            target_ = target.view_as(pred)
            correct += pred.eq(target_).sum().item()
            
            delta = (1 - target_.eq(pred)).detach().cpu().numpy()
            err = (target_ + 1).detach().cpu().numpy() * delta
            errors.append(err)

    errors = np.concatenate(errors)
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    print ([(errors==(i+1)).sum() for i in range(8)])
    return acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default="circle")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        dataset=CorpusDataset(f"data/{args.dataset}/train.npz", True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        dataset=CorpusDataset(f"data/{args.dataset}/test.npz", False),
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
        torch.save(model.state_dict(),"model.pt")
        
if __name__ == '__main__':
    main()