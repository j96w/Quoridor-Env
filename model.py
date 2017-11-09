from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init


class agentNET(torch.nn.Module):
    def __init__(self, num_inputs = 1, num_outputs = 150):
        super(agentNET, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 12, 10, stride=1, padding=0)
        self.conv2 = nn.Conv2d(12, 48, 4, stride=1, padding=0)
        self.conv3 = nn.Conv2d(48, 96, 4, stride=1, padding=0)
        self.conv4 = nn.Conv2d(96, 192, 2, stride=1, padding=0)

        self.lstm = nn.LSTMCell(1728, 864)

        self.fc1 = nn.Linear(864, 432)
        self.fc2 = nn.Linear(432, 216)

        self.critic_linear = nn.Linear(216, 1)
        self.actor_linear = nn.Linear(216, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.fc1.weight.data = norm_col_init(
            self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data = norm_col_init(
            self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = F.relu(self.fc1(hx))
        x = F.relu(self.fc2(x))

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)