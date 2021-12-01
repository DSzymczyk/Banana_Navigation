import random

import numpy as np
import torch
from torch import optim

from ReplayBuffer import ReplayBuffer
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, state_size, action_size, model_path=None, learning_rate=5e-4, gamma=0.99, tau=1e-3,
                 buffer_size=int(1e5), batch_size=64):
        self._state_size = state_size
        self._action_size = action_size

        self._replay_buffer = ReplayBuffer(buffer_size, batch_size)

        self._online_model = Model(state_size, action_size).to(device)
        self._target_model = Model(state_size, action_size).to(device)
        if model_path is not None:
            self._online_model.load_state_dict(torch.load(model_path))
            self._online_model.eval()
            self._target_model.load_state_dict(torch.load(model_path))
            self._target_model.eval()
        self.optimizer = optim.Adam(self._online_model.parameters(), lr=learning_rate)

        self._learning_rate = learning_rate
        self._gamma = gamma
        self._tau = tau
        self._start_eps = 0.5
        self._eps = self._start_eps
        self._eps_mul = 0.995
        self._min_eps = 0.01
        self._step = 0

        self._step_count = 0
        self._act_count = 0
        self._learn_count = 0
        self._soft_update_target_model_count = 0

    def __repr__(self):
        return f'Agent: [learning_rate={self._learning_rate}, gamma={self._gamma}, tau={self._tau}, eps: {self._start_eps}]'

    def step(self, state, action, reward, next_state, done):
        '''
            Add current step to replay buffer and learn every 4 steps
            :param state: current state
            :param action: current action
            :param reward: current reward
            :param next_state: next state
            :param done: boolean flag if episode is done
        '''
        self._step_count += 1
        self._replay_buffer.add(state, action, reward, next_state, done)
        self._step += 1
        if self._step % 4 == 0:
            self.learn()

    def choose_action(self, state, train=True):
        '''
        If train flag is set to True choose action in epsilon greedy policy otherwise choose best action selected by
        agent.
        :param state: current state
        :param train: flag checking if epsilon greedy policy should be applied
        :return: Action selected by agent
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self._online_model.eval()
        with torch.no_grad():
            action_values = self._online_model(state)
        self._online_model.train()

        if train and random.random() < self._eps:
            return random.choice(np.arange(self._action_size))
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self):
        '''
        Training model using Double DQN.
        '''
        sample = self._replay_buffer.get_sample()
        if sample is None:
            return
        states, actions, rewards, next_states, dones = sample
        argmax_online_model_action = self._online_model(next_states).max(1)[1]
        target_model_q_values = self._target_model(next_states).detach()
        q_values_per_argmax = target_model_q_values[np.arange(len(states)), argmax_online_model_action]
        q_values_per_argmax = q_values_per_argmax.unsqueeze(1)
        q_values_per_argmax *= (1 - dones)
        target_q_values = rewards + (self._gamma * q_values_per_argmax)

        q_values_estimates = self._online_model(states).gather(1, actions)

        loss = q_values_estimates - target_q_values
        loss = loss.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update_target_model()

    def soft_update_target_model(self):
        '''
        Updating target model using Polyak averaging.
        '''
        for target_param, local_param in zip(self._target_model.parameters(), self._online_model.parameters()):
            target_param.data.copy_(self._tau * local_param + (1 - self._tau) * target_param)

    def decrease_epsilon(self):
        '''
        Decrease epsilon by _eps_mul every function run. Minimal epsilon value is declared by _min_eps parameter.
        '''
        self._eps = max(self._eps * self._eps_mul, self._min_eps)
