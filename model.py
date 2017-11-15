from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init


class agentNET(torch.nn.Module):
    def __init__(self, num_inputs = 1, num_outputs = 129):
        super(agentNET, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 12, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(12, 48, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(48, 96, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(96, 96, 7, stride=1, padding=0)

        self.fc1 = nn.Linear(9600, 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, 200)

        self.critic_linear = nn.Linear(200, 1)
        self.actor_linear = nn.Linear(200, num_outputs)

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

        self.fc3.weight.data = norm_col_init(
            self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.critic_linear(x), self.actor_linear(x)