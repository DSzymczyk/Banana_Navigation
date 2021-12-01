# Banana Navigation
Purpose of this project is to solve Banana Unity Environment using Dueling DDQN algorithm. Environment will be 
considered solved when average score in last 100 episodes is equal or greater than 13. Project requires 64-bit Winsows
to run.

## Getting started:
1. Install required packages: `pip install -r requirements.txt`
2. Launch training: `python NavigationTrain.py`. 
   - If training solve the environment `saved_model.pth` file will be saved.
   - Training progress is displayed every 25 episodes.
3. Launch test: `python NavigationTrain.py`.
   - Test shows currently saved model in action.
   - Model is saved as `saved_model.pth` file.
   
## Trained model playthrough
![Alt Text](BananaNavigation.gif)