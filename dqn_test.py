import torch

from dqn import DQN
from env import Environment, Preprocessor

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'


def select_action(state):
    policy_net.eval()
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.

        return policy_net(state).max(1)[1].view(1, 1).long()
    
    
if __name__=='__main__':
    env = Environment(deploy=True)
    BATCH_SIZE = 32
    

    width = 80
    height = 80
    preprocessor = Preprocessor(width, height)
    
    # change fname here
    fname = 'final.pt'

    n_actions = len(env.actions.keys())
    policy_net = DQN(height, width, n_actions).float().to(device)
    policy_net.load_state_dict(torch.load(fname, map_location=device))
    
    
    

    episode_rewards = []
    episode_durations = []
    episode_losses = []
    steps_done = 0


    # Initialize the environment and state
    frame, _, done = env.start_game()
    frame = preprocessor.process(frame)
    state = preprocessor.get_initial_state(frame)
    state = torch.tensor(state).unsqueeze(0).float().to(device)
    cum_steps = 0
    step = 0
    global cum_loss
    cum_loss = 0
    cum_reward = 0

    
    while True:
        # Select and perform an action
        action = select_action(state)
        action_str = Environment.actions[action.item()]
        frame, reward, done = env.do_action(action.item())
        frame = preprocessor.process(frame)
        next_state = preprocessor.get_updated_state(frame)
        next_state = torch.tensor(next_state).unsqueeze(0).float().to(device)
        
        reward = torch.tensor([reward], device=device).float()
        cum_reward += reward
        cum_steps += 1

        # Move to the next state
        state = next_state
        
        step+=1
