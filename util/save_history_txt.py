import os
import datetime
import numpy as np

def save_history_txt(SAVE, device, path, Q_net, QnetToCell, env, i_episode, t):
    if i_episode % SAVE == 0:
        V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, device)
        print(f"--------{i_episode} is saved with {t} steps. V_table and Action table")
            
        if not os.path.isdir(path+'/V_table_train') or not os.path.isdir(path+'/Action_table_train'):
            os.makedirs(path+'/V_table_train')
            os.makedirs(path+'/Action_table_train')
            
        now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
        np.savetxt(f'{path}/V_table_train/V_table_{i_episode}_{now}.txt', V_table, fmt='%.3f')
        np.savetxt(f'{path}/Action_table_train/Action_table_{i_episode}_{now}.txt', Action_table, fmt='%d')