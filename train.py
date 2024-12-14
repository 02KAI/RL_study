import gymnasium as gym
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import logging
import hydra
from torch.distributions import Normal
from omegaconf import DictConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given layer.
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

class Actor(nn.Module):
    def __init__(self, envs):
        super(Actor, self).__init__()
        self.obs_dim = envs.single_observation_space.shape[0]
        self.action_dim = envs.single_action_space.shape[0]
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01)
        )
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        self.to(device)
        
    def forward(self, state):
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

class Critic(nn.Module):
    def __init__(self, envs):
        super(Critic, self).__init__()
        self.obs_dim = envs.single_observation_space.shape[0]
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.to(device)
    
    def forward(self, state):
        return self.net(state)

class PPO:
    def __init__(self, envs, cfg):
        self.envs = envs
        
        self.actor = Actor(self.envs)
        self.critic = Critic(self.envs)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.ppo.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.ppo.critic_lr)
        
        self.clip_param = cfg.ppo.clip_param
        self.max_grad_norm = cfg.ppo.max_grad_norm
        self.ppo_epochs = cfg.ppo.ppo_epochs
        self.batch_size = cfg.ppo.batch_size
        self.gamma = cfg.ppo.gamma
        self.gae_lambda = cfg.ppo.gae_lambda
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().detach().numpy(), action_log_prob.cpu().detach().numpy()
    
    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        for _ in range(self.ppo_epochs):
            for index in range(0, len(states), self.batch_size):
                batch_states = states[index:index + self.batch_size]
                batch_actions = actions[index:index + self.batch_size]
                batch_log_probs = old_log_probs[index:index + self.batch_size]
                batch_returns = returns[index:index + self.batch_size]
                batch_advantages = advantages[index:index + self.batch_size]
                
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_pred = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(value_pred, batch_returns)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
        return actor_loss.item(), critic_loss.item()

@hydra.main(config_path=".", config_name="config")
def train(cfg: DictConfig):
    set_seed(42, deterministic=True)
    
    envs = gym.make_vec(cfg.env.name, num_envs=cfg.env.num_envs, render_mode="human" if cfg.train.render else None)
    agent = PPO(envs, cfg)
    
    num_episodes = cfg.train.num_episodes
    max_steps = cfg.train.max_steps
    
    pbar = tqdm(total=num_episodes, desc="Training Progress")
    
    all_rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        states, _ = envs.reset()
        episode_reward = np.zeros(cfg.env.num_envs)
        all_states, all_actions, all_log_probs, all_episode_rewards = [], [], [], []

        for _ in range(max_steps):
            if cfg.train.render:
                envs.render()
            
            actions, log_probs = agent.get_action(states)
            next_states, rewards, terminated, truncated, _ = envs.step(actions)
            
            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_episode_rewards.append(rewards)
            
            states = next_states
            episode_reward += rewards
            
            if np.any(terminated) or np.any(truncated):
                break
        
        all_states = np.concatenate(all_states)
        all_actions = np.concatenate(all_actions)
        all_log_probs = np.concatenate(all_log_probs)
        all_episode_rewards = np.concatenate(all_episode_rewards)
        
        returns = []
        advantages = []
        value = 0
        for reward in reversed(all_episode_rewards):
            value = reward + agent.gamma * value
            returns.insert(0, value)
        returns = np.array(returns)
        
        values = agent.critic(torch.FloatTensor(all_states).to(device)).cpu().detach().numpy()
        for i in range(len(all_episode_rewards)):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss, critic_loss = agent.update(all_states, all_actions, all_log_probs, returns, advantages)
        
        all_rewards.append(episode_reward.mean())
        avg_reward = np.mean(all_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        pbar.update(1)
        
        log.info(f"Episode {episode + 1}/{num_episodes}")
        log.info(f"Reward: {episode_reward.mean():.2f}")
        log.info(f"Average Reward: {avg_reward:.2f}")
        log.info(f"Actor Loss: {actor_loss:.4f}")
        log.info(f"Critic Loss: {critic_loss:.4f}")

    pbar.close()
    envs.close()
    
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, 'halfcheetah_model.pth')

if __name__ == "__main__":
    train()