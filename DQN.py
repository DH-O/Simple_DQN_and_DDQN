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

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

X_SIZE          = 10
Y_SIZE          = 10

STATE_DIM       = 2
ACTION_DIM      = 4

NUM_EPISODES    = 1000 # 
TIME_LIMIT      = 2*X_SIZE*Y_SIZE 
TEST_EPISODES   = 10

TARGET_UPDATE   = NUM_EPISODES//10

RM_SIZE             = 1000000
BATCH_SIZE          = 256
GAMMA               = 0.9
EPS_START           = 0.7
EPS_END             = 0.2
EPS_DECAY           = NUM_EPISODES
RANDOM_ACTION_PROB  = 0.1

SAVE            = NUM_EPISODES//10

#### STEPS DONE
steps_done = 0

#### Argument parser
parser = argparse.ArgumentParser(description='DQN')

parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

#### logging
now_day = datetime.datetime.now().strftime("%m-%d")
now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")

path = f'./results_DQN/{now_day}_{X_SIZE, Y_SIZE}_{GAMMA}_NE_{NUM_EPISODES}_TSL_{TIME_LIMIT}_{TARGET_UPDATE}_{RANDOM_ACTION_PROB}_{RM_SIZE}/result_{now}'
writer = SummaryWriter(f'{path}/tensorboard_{now}')

#### Networks
Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to(device)
target_Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to(device)
target_Q_net.load_state_dict(Q_net.state_dict())
target_Q_net.eval()

#### optimizer
optimizer = optim.RMSprop(Q_net.parameters())   # Optimizer should only work in Q_net
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
        eps_threshold = np.clip(eps_threshold, RANDOM_ACTION_PROB, 0.9)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return Q_net(state).max(0)[1]
        else:
            return torch.tensor(random.randrange(ACTION_DIM), device=device, dtype=torch.int64)

def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
    
    batch = buffer.Transition(*zip(*transitions))
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action).unsqueeze(-1)
    reward_batch = torch.stack(batch.reward)
    non_final_next_states = torch.stack(batch.next_state)
    mask_batch = torch.stack(batch.mask)
    
    Q_values = Q_net(state_batch).gather(1, action_batch)   # q(s,a)
    
    with torch.no_grad():
        next_state_Q_values_array = target_Q_net(non_final_next_states).max(1)[0]                       # max_a(q_target(s',a))
        expected_Q_values_array = (next_state_Q_values_array.mul(mask_batch) * GAMMA) + reward_batch    # r + gamma * max_a(q_target(s',a)) * mask_batch
    
    criterion = nn.MSELoss()
    loss = criterion(Q_values, expected_Q_values_array.unsqueeze(-1))
    
    optimizer.zero_grad()               # optimizer reset
    loss.backward()                     # calculate backprop
    for param in Q_net.parameters():
        param.grad.data.clamp_(-1, 1)   # clamp parameters of the network
    optimizer.step()                    # apply backprop
    
    return loss

if __name__ == '__main__':
    
    env = simple_gym.TwoDimCoordinationMap(X_SIZE, Y_SIZE)
    env.SimpleAntMazation()
    
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
        
        state = env.reset()
        ### reward_states_reset check
        np.savetxt(f'{path}/SimpleMaze_Reward_table_reset.txt', env.reward_states, fmt='%d')
        
        state = torch.tensor(state, device=device, dtype=torch.float32)
        
        for t in range(1, TIME_LIMIT+1): # t = 1 ~ TIME_LIMIT
            action = select_action(state, test=False)
            next_state, reward, done = env.step(action.item())  # action.item() are the pure values from the tensor
            reward = torch.tensor(reward,dtype=torch.float32 ,device=device)
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
            
            buffer.push(state, action, next_state, reward, torch.tensor(1-int(done), device=device, dtype=torch.float32))
            state = next_state
            
            loss = optimize_model()
            
            if steps_done % TARGET_UPDATE == 0:
                target_Q_net.load_state_dict(Q_net.state_dict())
            if done:
                break
            
        if loss is not None:
            writer.add_scalar('success_rate/train', int(done), i_episode)
            writer.add_scalar('steps_per_episode/train', t, i_episode)
            writer.add_scalar('Loss/train', loss, i_episode)
        
        if i_episode % 10 == 0:
            print(f"episode of {i_episode} is done with {t} steps. It should be less than {X_SIZE*Y_SIZE}")
            
            #### test ####
            done_stack = 0
            steps_stack = 0
            
            for test_episode in range(1, TEST_EPISODES+1):
                state = env.reset()
                state = torch.tensor(state, device=device, dtype=torch.float32)
                
                for test_t in range(1, TIME_LIMIT+1):
                    action = select_action(state, test=True)
                    next_state, _, done = env.step(action.item())
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
                    state = next_state
                    if done:
                        break
                done_stack += int(done)
                steps_stack += test_t
                
                writer.add_scalar('success_rate/test', done_stack/TEST_EPISODES, i_episode)
                writer.add_scalar('steps_per_episode/test', steps_stack/TEST_EPISODES, i_episode)
                V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, device)
        
                if not os.path.isdir(path+'/V_table_test') or not os.path.isdir(path+'/Action_table_test'):
                        os.makedirs(path+'/V_table_test')
                        os.makedirs(path+'/Action_table_test')
                
                now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
                np.savetxt(f'{path}/V_table_test/V_table_test_{now}.txt', V_table, fmt='%.3f')
                np.savetxt(f'{path}/Action_table_test/Action_table_test_{now}.txt', Action_table, fmt='%d')
        
        if i_episode % SAVE == 0:
            V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, device)
            print(f"--------{i_episode} is saved with {t} steps. V_table and Action table")
            
            if not os.path.isdir(path+'/V_table_train') or not os.path.isdir(path+'/Action_table_train'):
                os.makedirs(path+'/V_table_train')
                os.makedirs(path+'/Action_table_train')
            
            now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
            np.savetxt(f'{path}/V_table_train/V_table_{i_episode}_{now}.txt', V_table, fmt='%.3f')
            np.savetxt(f'{path}/Action_table_train/Action_table_{i_episode}_{now}.txt', Action_table, fmt='%d')    

    writer.close()
    print(f'{path} is done')
    print('Complete')