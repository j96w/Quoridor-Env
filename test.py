from __future__ import division
import torch
import torch.nn.functional as F
from utils import setup_logger
from model import agentNET
from torch.autograd import Variable
from Quoridor import Quoridor
import time
import logging
import random
import numpy as np


def test(args, shared_model):
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    env = Quoridor(15)

    model = agentNET(1, 150)
    model.eval()

    start_time = time.time()
    success_time = 0
    ctime = 0
    max_time = 0

    while True:
        success_time = 0
        model.load_state_dict(shared_model.state_dict())
        if args.gpu:
            model = model.cuda()

        for i in range(100):
            if args.gpu:
                cx = Variable(torch.zeros(1, 864).cuda())
                hx = Variable(torch.zeros(1, 864).cuda())
            else:
                cx = Variable(torch.zeros(1, 864))
                hx = Variable(torch.zeros(1, 864))

            state, _ = env.reset()
            state = torch.from_numpy(np.array([state, ])).float()


            for step in range(500):
                if args.gpu:
                    value, logit, (hx, cx) = model((Variable(state.unsqueeze(0).cuda()),(hx, cx)))
                else:
                    value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),(hx, cx)))

                prob = F.softmax(logit)
                if args.gpu:
                    action = prob.max(1)[1].data.cpu()
                else:
                    action = prob.max(1)[1].data
                # print(action)
                # print(action.numpy())
                # print(action.numpy().tolist())
                # print(action.numpy().tolist()[0][0])
                state, result = env.action(action.numpy().tolist()[0] - 1)
                state = torch.from_numpy(np.array([state, ])).float()

                if(result == 0):
                    success_time += 1

                if(result == 2):
                    done = False
                else:
                    done = True

                if done:
                    break

        success_mean = float(success_time) / 100

        log['{}_log'.format(args.env)].info(
                "Time {0}, success mean {1:.5f}, success time {2}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    success_mean, success_time))

        if args.gpu:
            model.cpu()

        if success_time >= max_time:
            model.load_state_dict(shared_model.state_dict())
            state_to_save = model.state_dict()
            torch.save(state_to_save, '{0}{1}_agent.dat'.format(args.save_model_dir, args.env))

            max_time = success_time

        time.sleep(120)
