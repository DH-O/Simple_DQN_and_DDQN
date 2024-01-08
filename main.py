# code source from https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html

import os
import math
import random
import argparse
import datetime
import numpy as np

import env.simple_gym as simple_gym
import util.utils as utils
import model.Networks as Networks

from util.ReplayMemory import ReplayMemory
from train import train
from test import test
from util.save_history_txt import save_history_txt
from model.DQN import optimize_model_DQN
from model.DDQN import optimize_model_DDQN

import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

X_SIZE          = 6
Y_SIZE          = 6

STATE_DIM       = 2
ACTION_DIM      = 4

NUM_EPISODES    = 1000 # 
TIME_LIMIT      = 2*X_SIZE*Y_SIZE 
TEST_EPISODES   = 10

TARGET_UPDATE   = NUM_EPISODES//10

RM_SIZE             = 1000000
BATCH_SIZE          = 2048
GAMMA               = 0.9
EPS_START           = 0.7
EPS_END             = 0.15
EPS_DECAY           = NUM_EPISODES

SAVE            = NUM_EPISODES//10

#### STEPS DONE
steps_done = 0

#### Argument parser
parser = argparse.ArgumentParser()

#### device
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--model', type=str, default='DQN', help='DQN or DDQN')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

#### logging
now_day = datetime.datetime.now().strftime("%m-%d")
now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")

path = f'./results_{args.model}/{now_day}_{X_SIZE, Y_SIZE}_GAM_{GAMMA}_NE_{NUM_EPISODES}_TE_{TARGET_UPDATE}_END_{EPS_END}_{RM_SIZE}_BS_{BATCH_SIZE}/result_{now}'
writer = SummaryWriter(f'{path}/tensorboard_{now}')

#### Networks
Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to(device)
target_Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to(device)
target_Q_net.load_state_dict(Q_net.state_dict())
target_Q_net.eval()

#### optimizer
optimizer = optim.Adam(Q_net.parameters())   # Optimizer should only work in Q_net
buffer = ReplayMemory(RM_SIZE)
QnetToCell = utils.QnetToCell(X_SIZE, Y_SIZE)

def select_action(state, test):
    global steps_done
    if test:
        return Q_net(state).max(0)[1]
    else:
        sample = random.random()    # random value b/w 0 ~ 1 
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return Q_net(state).max(0)[1]
        else:
            return torch.tensor(random.randrange(ACTION_DIM), device=device, dtype=torch.int64)

def optimize_model():
    if args.model == 'DQN':
        return optimize_model_DQN(buffer, BATCH_SIZE, Q_net, target_Q_net, optimizer, GAMMA)
    else:
        return optimize_model_DDQN(buffer, BATCH_SIZE, Q_net, target_Q_net, optimizer, GAMMA)

if __name__ == '__main__':
    
    env = simple_gym.TwoDimArrayMap(X_SIZE, Y_SIZE)
    env.SimpleMazation()
    
    #### Maze logging
    if not os.path.isdir(path):
        os.makedirs(path)
    np.savetxt(f'{path}/SimpleMaze_table.txt', env.maze, fmt='%d')
    np.savetxt(f'{path}/SimpleMaze_Reward_table.txt', env.reward_states, fmt='%d')
    print(f'{path} is running...')
    
    #### Training
    done_stack = 0
    steps_stack = 0
    
    for i_episode in range(1, NUM_EPISODES+1):
        
        t, done, loss = train(TIME_LIMIT, TARGET_UPDATE, steps_done, device, path, Q_net, target_Q_net, buffer, select_action, optimize_model, env)
        
        ### tensorboard    
        if loss is not None:
            writer.add_scalar('success_rate/train', int(done), i_episode)
            writer.add_scalar('steps_per_episode/train', t, i_episode)
            writer.add_scalar('Loss/train', loss, i_episode)
        
        test(X_SIZE, Y_SIZE, TIME_LIMIT, TEST_EPISODES, device, path, writer, Q_net, QnetToCell, select_action, env, i_episode, t)
        
        save_history_txt(SAVE, device, path, Q_net, QnetToCell, env, i_episode, t)    

    writer.close()
    print(f'{path} is done')
    print('Complete')