import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import time
import argparse

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.to(device)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.to(device)
    
    def forward(self, state):
        return self.net(state)

class PPO:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().detach().numpy()[0], action_log_prob.cpu().detach().item()
    
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

def train(render=False):
    # 根据render参数选择是否启用渲染
    env = gym.make("HalfCheetah-v4", render_mode="human" if render else None)
    agent = PPO(env)
    num_episodes = 30000
    max_steps = 3000
    
    pbar = tqdm(total=num_episodes, desc="Training Progress")
    
    all_rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states, actions, log_probs, rewards = [], [], [], []
        
        step_bar = tqdm(total=max_steps, desc=f"Episode {episode+1}", leave=False)
        
        for step in range(max_steps):
            # 仅在render=True时渲染
            if render:
                env.render()
            
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            
            step_bar.update(1)
            
            if terminated or truncated:
                break
        
        step_bar.close()
        
        returns = []
        advantages = []
        value = 0
        for reward in reversed(rewards):
            value = reward + agent.gamma * value
            returns.insert(0, value)
        returns = np.array(returns)
        
        values = agent.critic(torch.FloatTensor(states).to(device)).cpu().detach().numpy()
        for i in range(len(rewards)):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss, critic_loss = agent.update(states, actions, log_probs, returns, advantages)
        
        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        pbar.update(1)
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'avg_reward': f'{avg_reward:.2f}',
            'actor_loss': f'{actor_loss:.4f}',
            'critic_loss': f'{critic_loss:.4f}'
        })
        
    pbar.close()
    env.close()
    
    # 保存模型
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, 'halfcheetah_model.pth')

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train HalfCheetah with PPO')
    parser.add_argument('--render', action='store_true', 
                      help='Enable rendering (default: False)')
    args = parser.parse_args()
    
    # 使用命令行参数来控制渲染
    train(render=args.render)