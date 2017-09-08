from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale(size=224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Scale(size=224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='/app/pytorch/data', train=False, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

testset = datasets.CIFAR10(root='/app/pytorch/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.layers = [nn.Conv2d(3, 64, kernel_size=3, padding=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(64, 64, kernel_size=3, padding=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Conv2d(64, 128, kernel_size=3, padding=1),
                       nn.BatchNorm2d(128),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(128, 128, kernel_size=3, padding=1),
                       nn.BatchNorm2d(128),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Conv2d(128, 256, kernel_size=3, padding=1),
                       nn.BatchNorm2d(256),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(256, 256, kernel_size=3, padding=1),
                       nn.BatchNorm2d(256),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(256, 256, kernel_size=3, padding=1),
                       nn.BatchNorm2d(256),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(256, 256, kernel_size=3, padding=1),
                       nn.BatchNorm2d(256),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Conv2d(256, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.AvgPool2d(kernel_size=1, stride=1)
                       ]
        self.features = nn.Sequential(*self.layers)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = VGG19()
print(net)

if args.cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        time_elapsed = time.time() - start
        img_sec = args.batch_size / time_elapsed
        print('Batch %d / %d | Loss: %.3f | Acc: %.3f%% (%d/%d) | %.3f img/sec'
              % (batch_idx, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                 img_sec))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print('Batch %d / %d Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (batch_idx, len(testloader), test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
