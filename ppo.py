from env import Environment, Preprocessor

import math
import random
import numpy as np
from collections import namedtuple
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class PPONetwork(nn.Module):
    def __init__(self, h, w, n_actions):
        super(PPONetwork, self).__init__()
        
        # Shared convolutional layers
        self.shared_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64, 96, kernel_size=3, stride=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
        )
        
        # Calculate output size of convolutional layers
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
            
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        
        linear_input_size = convw * convh * 96
        self.linear_input_size = linear_input_size
        
        # Shared fully connected layers
        self.fc_shared = nn.Sequential(
            layer_init(nn.Linear(linear_input_size, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh()
        )
        
        # Policy head (actor)
        self.policy_head = layer_init(nn.Linear(256, n_actions))
        
        # Value head (critic)
        self.value_head = layer_init(nn.Linear(256, 1))
    
    def forward(self, x):
        x = x.float()
        x = self.shared_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        
        # Actor output (policy)
        policy_logits = self.policy_head(x)
        
        # Critic output (value)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_policy(self, x):
        policy_logits, _ = self.forward(x)
        return policy_logits
    
    def get_value(self, x):
        _, value = self.forward(x)
        return value

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        states = torch.cat(self.states)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=device)
        probs = torch.tensor(self.probs, dtype=torch.float, device=device)
        values = torch.tensor(self.values, dtype=torch.float, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float, device=device)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=device)
        
        return states, actions, probs, values, rewards, dones, batches

class PPOTrainer:
    def __init__(self, actor_critic, lr=1e-4, gamma=0.99, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=32, n_epochs=10, entropy_coef=0.01,
                 value_coef=0.5):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.memory = PPOMemory(batch_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
    
    def store_transition(self, state, action, prob, value, reward, done):
        self.memory.store(state, action, prob, value, reward, done)
    
    def select_action(self, observation):
        self.actor_critic.eval()
        
        with torch.no_grad():
            policy_logits, value = self.actor_critic(observation)
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            prob = dist.log_prob(action)
        
        return action.item(), prob.item(), value.item()
    
    def learn(self):
        self.actor_critic.train()
        
        for _ in range(self.n_epochs):
            states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
            
            advantages = self._compute_advantages(rewards, values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_values = values[batch_indices]
                batch_rewards = rewards[batch_indices]
                
                policy_logits, new_values = self.actor_critic(batch_states)
                
                # Get new action probabilities
                dist = Categorical(logits=policy_logits)
                new_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate policy loss (Actor loss)
                ratio = torch.exp(new_probs - batch_old_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss (Critic loss)
                returns = batch_advantages + batch_values
                value_loss = F.mse_loss(new_values.squeeze(-1), returns)
                
                # Calculate total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Perform optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear()
    
    def _compute_advantages(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t].item()
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages[t] = last_advantage
        
        return advantages

if __name__ == "__main__":
    env = Environment(debug=True)
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
    actor_critic = PPONetwork(height, width, n_actions).float().to(device)
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
    # actor_critic.load_state_dict(torch.load("meta/ppo_model_ep_X_reward_Y.pt", map_location=device))
    
    episode_rewards = []
    episode_durations = []
    episode_losses = []
    steps_done = 0
    num_episodes = 100000
    update_interval = 2048  # PPO typically uses large update intervals
    time_steps = 0
    
    for i_episode in range(num_episodes):
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
            
            # Learn if we've collected enough transitions
            if time_steps % update_interval == 0:
                ppo_trainer.learn()
        
        if cum_steps > 0:
            episode_durations.append(cum_steps)
            episode_rewards.append(cum_reward)
            print(f"Episode {i_episode}: duration: {cum_steps}, reward: {cum_reward}")
        
        # Save weights
        if i_episode % 10 == 0:
            torch.save(actor_critic.state_dict(), f'meta/ppo_model_ep_{i_episode}_reward_{cum_steps}.pt')
    
    with open("episode_durations.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(episode_durations, indent=4, ensure_ascii=False))
    with open("episode_rewards.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(episode_rewards, indent=4, ensure_ascii=False))
    
    print('Complete')
    savename = "final"
    torch.save(actor_critic.state_dict(), f'{savename}.pt')
