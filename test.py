import os
import torch
import datetime
import numpy as np

def test(X_SIZE, Y_SIZE, TIME_LIMIT, TEST_EPISODES, device, path, writer, Q_net, QnetToCell, select_action, env, i_episode, t):
    if i_episode % 100 == 0:
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