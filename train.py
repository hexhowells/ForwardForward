import torch
import matplotlib.pyplot as plt
from random import randint

from utils import *
from model import Model
import hyperparameters as hp


def generate_data(x, y):
    x_pos = merge_label(x, y)  # positive image

    rnd_idx = torch.randperm(y.size(0))
    
    # changing negative labels that are the same as its positive label
    y_ = y.clone()[rnd_idx]
    for i in range(rnd_idx.shape[0]):
        if y[i] == y_[i]:
            if y_[i] <= 5: y_[i] += randint(1, 4)
            else: y_[i] -= randint(1, 5)

    x_neg = merge_label(x, y_[rnd_idx])  # negative image

    return x_pos, x_neg


if __name__ == "__main__":
    torch.manual_seed(hp.seed)

    train_loader, test_loader = MNIST_loaders()

    model = Model([784, 200])

    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos, x_neg = generate_data(x, y)
    

    for epoch in range(hp.epochs):
        print(f'Training epoch {epoch}')
        model.train(x_pos, x_neg)

    print('train error:', 1.0 - model.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    print('test error:', 1.0 - model.predict(x_te).eq(y_te).float().mean().item())