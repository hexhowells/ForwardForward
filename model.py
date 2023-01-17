import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from tqdm import tqdm

import utils
import hyperparameters as hp


class Model(torch.nn.Module):
    def __init__(self, dims, device='cuda'):
        super().__init__()

        self.layers = []
        for d in range(len(dims)-1):
            layer = nn.Sequential(
                        utils.LPNormalise(),
                        nn.Linear(dims[d], dims[d+1]),
                        nn.ReLU()
                    ).to(device=device)
            self.layers.append(layer)


    def predict(self, x):
        goodness_per_label = []

        # for each label, merge with input and record the sum of the goodness for each layer
        # image|label pair with the best score is the prediction
        for label in range(10):
            h = utils.merge_label(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [utils.goodness(h)]

            goodness_per_label += [sum(goodness).unsqueeze(1)]

        goodness_per_label = torch.cat(goodness_per_label, 1)

        return goodness_per_label.argmax(1)


    def train(self, x_pos, x_neg):
        # train one layer at a time
        for i, layer in enumerate(self.layers):
            print(f'  training layer {i}')
            opt = Adam(layer.parameters(), lr=hp.lr)
            for _ in tqdm(range(hp.layer_epochs)):
                g_pos = utils.goodness(layer.forward(x_pos))
                g_neg = utils.goodness(layer.forward(x_neg))

                loss = utils.loss_fn(g_pos, g_neg, hp.threshold)

                opt.zero_grad()
                loss.backward()
                opt.step()

            x_pos = layer.forward(x_pos).detach()
            x_neg = layer.forward(x_neg).detach()
