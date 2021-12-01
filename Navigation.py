from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

from agent import Agent

env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size, action_size, model_path='saved_model.pth')

env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]
score = 0
while True:
    action = int(agent.choose_action(state, train=False))
    env_info = env.step(action)[brain_name]
    agent.step(state, action, env_info.rewards[0], env_info.vector_observations[0], env_info.local_done[0])
    score += env_info.rewards[0]
    state = env_info.vector_observations[0]
    if env_info.local_done[0]:
        break
print("Score: {}".format(score))

env.close()
