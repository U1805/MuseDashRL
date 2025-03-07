import torch
from env import Environment, Preprocessor
from ppo import PPONetwork, PPOTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env = Environment(deploy=True)
    BATCH_SIZE = 32
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    POLICY_CLIP = 0.2
    N_EPOCHS = 4
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    LR = 1e-4
    TARGET_UPDATE = 10
    
    width = 80
    height = 80
    preprocessor = Preprocessor(width, height)
    
    n_actions = len(env.actions.keys())
    actor_critic = PPONetwork(height, width, n_actions)
    ppo_trainer = PPOTrainer(
        actor_critic=actor_critic,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        policy_clip=POLICY_CLIP,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF
    )
    
    # Uncomment to load from a previous model
    actor_critic.load_state_dict(torch.load("final.pt", map_location=device))
    
    steps_done = 0
    num_episodes = 100000
    update_interval = 2048  # PPO typically uses large update intervals
    time_steps = 0
    
    # Initialize the environment and state
    frame, _, done = env.start_game()
    frame = preprocessor.process(frame)
    state = preprocessor.get_initial_state(frame)
    state = torch.tensor(state).unsqueeze(0).float().to(device)
    cum_steps = 0
    cum_reward = 0
    
    while not done:
        # Select and perform an action
        action, prob, value = ppo_trainer.select_action(state)
        frame, reward, done = env.do_action(action)
        frame = preprocessor.process(frame)
        next_state = preprocessor.get_updated_state(frame)
        next_state = torch.tensor(next_state).unsqueeze(0).float().to(device)
        
        # Convert reward to tensor
        reward_tensor = torch.tensor(reward, device=device).float()
        cum_reward += reward
        cum_steps += 1
        
        # Store transition
        ppo_trainer.store_transition(state, action, prob, value, reward, done)
        time_steps += 1
        
        # Move to the next state
        state = next_state
