# Report
Learning algorithm used to solve environment is Dueling Double DQN.

Double DQN uses two estimators for calculations. First to select index of best action to take in current state and 
second to get value of that action. This procedure is mitigating problem of overestimation. Normal DQN always select 
max of estimated action values, which might be incorrect because estimated action values are often noisy. This leads to
overestimation and lower performance.

Dueling DQN is an improvement that applies only to the network architecture. Architecture contains two estimators. First
estimator for calculating state-value function `V(s)` which contains only one node. Second estimator for calculating 
action-advantage function `A(s, a)` which contains one node per action. All layers before split are shared by both 
estimators. Output layer `Q function` has same number of nodes as action-advantage estimator. Output layer is merging 
state-value function node and action-advantage function node for corresponding action.

### Adjustable hyper parameters in algorithm contains:  
  - `learning rate` - learning rate for optimizer, determines step while minimizing loss function.
  - `gamma` - discount value for rewards. Determines how much past rewards are valued. Used for calculating q value.
  - `tau` - used for updating target model. Determines update rate from online model.
  - `buffer size` - maximum size of replay buffer. Determines how many experiences are stored at once.
  - `batch size` - size of batch used for training. Determines size of experiences batch used for learning.
  - `epsilon` - start value of epsilon used in epsilon greedy policy.

### Adjusting hyper parameters
Hyper parameters used during tests:
 - `learning_rate`: [0.0002, 0.0005, 0.0008]
 - `gamma`: [0.8, 0.9, 0.99, 0.999, 1]
 - `tau`: [0.01, 0.001, 0.0001]
 - `buffer_size`: [10000, 100000]
 - `batch_size`: [64, 128, 256]
 - `epsilon`: [0.5, 1.0]

Algorithm has the highest performance when hyper parameters are set to:
 - `learning_rate`:  5e-4
 - `gamma`: 1
 - `tau`: 0.001
 - `buffer_size`: 10000
 - `batch_size`: 64
 - `epsilon`: 0.5

Agent with such hyper parameters solve environment in 329 episodes.

Maximum average score in 1000 episodes achieved by agent is 16.56.

### Training Results
![Alt Text](report/training_results.png)

### Ideas for future work
 - Implementing Prioritized Experience Replay to improve algorithm performance
 - Refactor algorithm to use raw pixel data as input