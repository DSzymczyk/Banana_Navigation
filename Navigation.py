from collections import deque

from unityagents import UnityEnvironment
import numpy as np

from agent import Agent

env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size, action_size, 42)
episodes = 10000
scores_window = deque(maxlen=100)
max_score = 0

for episode_number in range(1, episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = int(agent.choose_action(state))
        env_info = env.step(action)[brain_name]
        agent.step(state, action, env_info.rewards[0], env_info.vector_observations[0], env_info.local_done[0])
        score += env_info.rewards[0]
        state = env_info.vector_observations[0]
        if env_info.local_done[0]:
            break
    agent.decrease_epsilon()
    scores_window.append(score)
    mean_score = np.mean(scores_window)
    if episode_number % 5 == 0:
        print("Episode: {} - Average: {:.2f}".format(episode_number, mean_score))
    if mean_score > 13:
        print(f'Finished in {episode_number} episodes!')
        break
    # if mean_score > max_score:
    #     max_score = mean_score
    #     print(f'Saving model after {episode_number} episodes with mean score: {mean_score}')
    #     torch.save(agent._online_model.state_dict(), 'checkpoint.pth')

env.close()
