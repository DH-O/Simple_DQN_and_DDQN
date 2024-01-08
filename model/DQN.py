import torch
import torch.nn as nn

def optimize_model_DQN(buffer, BATCH_SIZE, Q_net, target_Q_net, optimizer, GAMMA):
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