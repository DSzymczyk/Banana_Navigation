# Banana Navigation
 - Purpose of this project is to solve Banana Unity Environment using Dueling DDQN algorithm. 
 - The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects 
   around the agent's forward direction. 
 - The action space contains 4 actions: move forward, move backward, turn left and turn right.
 - Environment will be considered solved when average score in last 100 episodes is equal or greater than 13. 
 - Project requires 64-bit Windows to run.

## Getting started:
1. Install required packages: `pip install -r requirements.txt`
2. Launch training: `python NavigationTrain.py`. 
   - If training solve the environment `saved_model.pth` file will be saved.
   - Training progress is displayed every 25 episodes.
3. Launch test: `python Navigation.py`.
   - Test shows currently saved model in action.
   - Model is saved as `saved_model.pth` file.
   
## Accreditation
Dueling Double DQN algorithm was written based on `Grokking Deep Reinforcement Learning` by Miguel Morales. 
   
## Trained model playthrough
![Alt Text](BananaNavigation.gif)