import torch.nn as nn

class QNET(nn.Module):

    def __init__(self, input_size, output_size):
        super(QNET, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))
        x = self.LReLU(self.fc3(x))
        return x