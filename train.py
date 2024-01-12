import torch
import numpy as np

def train(TIME_LIMIT, TARGET_UPDATE, steps_done, device, path, Q_net, target_Q_net, buffer, select_action, optimize_model, env):
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
            q_target_state_dict = target_Q_net.state_dict()
            q_state_dict = Q_net.state_dict()
            for key in q_state_dict:    # key is the name of the layer. ex) fc1.weight
                q_target_state_dict[key] = q_state_dict[key]*0.5 + q_target_state_dict[key]*(1-0.5) # 0.005 is the tau value, soft update
            target_Q_net.load_state_dict(q_target_state_dict)
        if done:
            break
    return t,done,loss