from torch.nn import Module

class Accuracy(Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred, target):
        pred = pred.argmax(dim=1)
        target = target.argmax(dim=1)
        return ((pred == target).sum().float() / float(target.size(0))).item()
