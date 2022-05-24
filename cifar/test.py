from re import L
import torch
from model import *
from torch.utils.data import ConcatDataset, DataLoader
import time
import sys

epochs = int(sys.argv[1]) # number of rounds
data_num = int(sys.argv[2]) # batches of data >= 1

test_data = torch.load('testing/test.pt')
train_data_arr = []
for i in range(0, data_num):
    t = torch.load('training/train' + str(i) +'.pt')
    train_data_arr.append(t)

train_data = ConcatDataset(train_data_arr)
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32)

start = time.time()
net  = Net()
i = train(net, trainloader, epochs, testloader, start)
end = time.time()
cross, acc = test(net, testloader)


print('time: {}, accuracy {}'.format(end - start, acc))