import gymnasium as gym
import torch
import numpy as np
import argparse
from torch.distributions import Normal
import torch.nn as nn

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

def evaluate_policy(model_path, render=True, num_episodes=10):
    # 创建环境
    env = gym.make("HalfCheetah-v4", render_mode="human" if render else None)
    
    # 创建Actor网络并加载模型
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # 获取动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mean, std = actor(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                action = action.cpu().numpy()[0]
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                done = True
            
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    # 打印统计信息
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HalfCheetah Policy')
    parser.add_argument('--model', type=str, default='halfcheetah_model.pth',
                      help='Path to the model file (default: halfcheetah_model.pth)')
    parser.add_argument('--render', action='store_true',
                      help='Enable rendering (default: False)')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate (default: 10)')
    
    args = parser.parse_args()
    
    evaluate_policy(
        model_path=args.model,
        render=args.render,
        num_episodes=args.episodes
    )