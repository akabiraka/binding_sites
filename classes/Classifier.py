
import math
import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes) -> None:
        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Linear(embed_dim, n_classes)

    def forward(self, seq_rep):
        # seq_rep: seq_len, embed_dim
        out = self.classifier(seq_rep) # shape: seq_len, n_classes
        return out

# x = torch.rand(10, 5)
# m = LinearClassifier(5, 1)
# print(m(x))


class WindowClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes, window_size=None) -> None:
        super(WindowClassifier, self).__init__()
        window_size = window_size if isinstance(window_size, int) else 9
        self.n_classes = 1 if n_classes <= 2 else n_classes
        self.conv = torch.nn.Conv2d(1, self.n_classes, kernel_size=(window_size, embed_dim), padding=(math.floor(window_size/2), 0)) # each out_channel is for each class

    def forward(self, seq_rep):
        # seq_rep: seq_len, embed_dim
        seq_rep.unsqueeze_(0)
        out = self.conv(seq_rep)
        out.squeeze_(2).t_()
        if self.n_classes<=2:
            return torch.sigmoid(out)  # for binary classification, use sigmoid for inference.
        else:
            return torch.softmax(out, dim=1)

# x = torch.rand(10, 5)
# m = WindowClassifier(5, 1, 3)
# print(m(x))


class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes) -> None:
        super(LSTMClassifier, self).__init__()
        self.n_classes = 1 if n_classes <= 2 else n_classes
        self.lstm = torch.nn.LSTM(embed_dim, n_classes, 1, batch_first=True, bidirectional=False) # embed_dim, n_classes
    
    def forward(self, seq_rep):
        # seq_rep: seq_len, embed_dim
        seq_rep.unsqueeze_(0)
        out, (hn, cn) = self.lstm(seq_rep)
        out.squeeze_(0)
        if self.n_classes<=2:
            return torch.sigmoid(out)  # for binary classification, use sigmoid for inference.
        else:
            return torch.softmax(out, dim=1) # for multiclass

# x = torch.rand(10, 5)
# m = LSTMClassifier(5, 4)
# print(m(x))



def classifier_factory(classfier_type, embed_dim, n_classes, window_size=None):
    if classfier_type=="linear":
        return LinearClassifier(embed_dim, n_classes)
    elif classfier_type=="window":
        print("Default window-size 9 is applied")
        return WindowClassifier(embed_dim, n_classes, window_size)
    elif classfier_type=="lstm":
        return LSTMClassifier(embed_dim, n_classes)

    else:
        raise NotImplementedError(f"classfier_type={classfier_type} not implemented.")