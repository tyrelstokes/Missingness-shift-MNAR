
import torch.nn as nn

class DenseGaussianNet(nn.Module):
    ''' this could be used as a probabilistic encoder as well as decoder '''
    def __init__(self, layers):
        super(DenseGaussianNet, self).__init__()
        net = []
        for _in, _out in zip(layers[:-2], layers[1:-1]):
            net.append(nn.Linear(_in, _out))
            net.append(nn.BatchNorm1d(_out))
            net.append(nn.ReLU())

        self.emb = nn.Sequential(*net)
        self.out_mu = nn.Linear(layers[-2], layers[-1])
        self.out_log_sigma = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        w = self.emb(x)
        return self.out_mu(w), self.out_log_sigma(w)
