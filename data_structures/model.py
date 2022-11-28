from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from torch.nn import Module
import torch
from torch import nn


dtype = torch.float32
device = torch.device("cuda:0")

len_rnn_seq = 3
near_by_vehicles_limit = 8
near_by_peds_limit = 5
near_by_touching_vehicles_limit = 5
near_by_touching_peds_limit = 5


class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.net.parameters():  # Imagenet results used in the first few conv layer
            param.requires_grad = False
        for param in self.net.layer4.parameters():  # the last convolutional layer (weights) is trainable
            param.requires_grad = True
        self.net.fc = nn.Linear(2048 * 2 * 2, 256)  # last conv layer shape (-1,2048,2,2)
        info_len = 9 + 5 * 3 + (near_by_peds_limit + near_by_touching_peds_limit +
                                near_by_touching_vehicles_limit + near_by_vehicles_limit) * 3
        self.fc2 = nn.Linear(256 + info_len, 128)

    def forward(self, input_tensor, info):
        out = self.net.forward(input_tensor)
        info = info.view(info.size()[0], -1)
        out = torch.cat([out, info], dim=1)  # faccio cat up sulle conv features e sui game data
        out = F.relu(self.fc2(out)) # returned feature shape (-1,128)

        return out


class RNNResnet(Module): #class that will contain my resnet, lstm, nn
    def __init__(self):
        super(RNNResnet, self).__init__()
        self.resnet = ResNet() #
        self.lstm = nn.LSTM(128, 64, batch_first=True) #per la policy
        self.action_fc3 = nn.Linear(64, 6)
        self.state_fc3 = nn.Linear(64, 1)

    def forward(self, inputs, infoes):
        features = self.resnet.forward(inputs, infoes)
        if len(features) >= len_rnn_seq:
            assert len(features) % len_rnn_seq == 0
            features = features.view(
                (len(features) // len_rnn_seq, -1)) # da ricontrollare, qualcosa non va
            out, _ = self.lstm(features)
        else:
            features = features.view(1, features.size()[0], features.size()[1])
            out, _ = self.lstm(features)

        out = torch.chunk(out, len_rnn_seq-1, dim=1)[-1]
        out = torch.squeeze(out, 1)
        policy = F.softmax(self.action_fc3(out), dim=1)
        state = self.state_fc3(out)
        return policy, state