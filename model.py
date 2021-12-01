import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_output = nn.Linear(64, 1)
        self.advantage_output = nn.Linear(64, action_size)

    def forward(self, state):
        '''
        Calculating scores for each action based by Dueling DQN. Argmax of this tensor is action picked by model.
        :param state:
        :return: Tensor with scores for each action.
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.value_output(x)
        a = self.advantage_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q
