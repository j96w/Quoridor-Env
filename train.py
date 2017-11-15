from __future__ import division
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import ensure_shared_grads
from model import agentNET
from torch.autograd import Variable
from shared_optim import SharedRMSprop, SharedAdam
import torch.nn as nn
import time
import random
import numpy as np
import torch.nn as nn
import copy
from utils import setup_logger
from Quoridor import Quoridor
import logging

def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)

    env = Quoridor(rank)
    model = agentNET(1, 129)

    model.train()
    criterion = nn.CrossEntropyLoss()

    done = True
    episode_length = 0
    uploadtime = 0
    before = 0

    while True:
        model.load_state_dict(shared_model.state_dict())

        state, _, opp_state, opp_action, _ = env.reset()
        state = torch.from_numpy(np.array([state, ])).float()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        opp_data = []
        if(opp_action != -1):
            opp_data.append(copy.deepcopy([opp_state, opp_action]))

        for step in range(args.num_steps):
            value, logit = model((Variable(state.unsqueeze(0))))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            action.view(-1, 1)
            log_prob = log_prob.gather(1, Variable(action))
            # print(action.numpy().tolist()[0])
            
            state, result, opp_state, opp_action = env.action(action.numpy().tolist()[0][0])
            state = torch.from_numpy(np.array([state, ])).float()

            if(opp_action != -1):
                opp_data.append(copy.deepcopy([opp_state, opp_action]))

            if result == 0:
                reward = 1
                done = True
            elif result == 2:
                dis0, _ = env.findPath(0)
                dis1, _ = env.findPath(1)
                reward = 0
                if (action.numpy().tolist()[0][0] < 128):
                    ans = float(dis1 - dis0 - before) / 5
                    if(ans > 0):
                        reward += ans
                before = dis1 - dis0
                done = False
            elif result == 1:
                reward = -1
                done = True
            else:
                reward = -2
                done = True

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                before = 0
                state, _, _, _, _ = env.reset()
                state = torch.from_numpy(np.array([state, ])).float()

        R = torch.zeros(1, 1)
        if not done:
            value, _ = model((Variable(state.unsqueeze(0))))
            R = value.data

        
        values.append(Variable(R))
        R = Variable(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]


        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        count = 0
        for step in range(args.num_steps):
            inputs = []
            labels = []
            optimizer.zero_grad()
            # print(opp_data)
            for i in range(10):
                tmp = random.randint(0, len(opp_data) - 1)
                inputs.append([copy.deepcopy(opp_data[tmp][0])])
                # print(inputs)
                # print(opp_data[tmp][0])
                # time.sleep(20)
                labels.append(copy.deepcopy(opp_data[tmp][1]))
            # print(len(inputs))
            # print(len(inputs[0]))
            # print(len(inputs[0][0]))
            # print(len(inputs[0][0][0]))
            # print(len(labels))
            # print(labels)
            inputs, labels = Variable(torch.FloatTensor(inputs)), Variable(torch.LongTensor(labels))
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()