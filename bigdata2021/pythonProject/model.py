import torch
import torch.nn as nn
from torchvision.models import resnet18

from torchsummary import summary

class Net(nn.Module):
    def __init__(self, n_cls=3, md=True, pretrained=True):
        super(Net, self).__init__()
        self.hidden_size = 200
        self.cnn = resnet18(pretrained)
        # self.cnn2 = resnet18(pretrained)
        self.md = md
        self.fcn = nn.Linear(1000*2 + 6 + 131*2, n_cls) if self.md else nn.Linear(1000*2 + 6, n_cls)

    def forward(self, x, iml, imr, d=None):
        x1 = self.cnn(iml)
        x2 = self.cnn(imr)
        # x: [b, 6], x1/x2: [b, 1000]
        if self.md:
            assert d is not None, 'when multi-mode, d must a 262-D tensor'
            x = torch.cat([x, x1, x2, d], dim=1)
        else:
            x = torch.cat([x, x1, x2], dim=1)
        y = self.fcn(x)
        return torch.sigmoid(y)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net().to(device)
    summary(net, (6, (3, 224, 224), (3, 224, 224)))
