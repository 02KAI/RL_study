import gymnasium as gym
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import logging
import hydra
import os
import wandb

from torch.distributions import Normal
from omegaconf import DictConfig
from datetime import datetime
from gymnasium.wrappers import RecordVideo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_gae(rewards, values, next_value, terminated, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE).
    """
    T, num_envs = rewards.shape
    advantages = np.zeros((T, num_envs), dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - terminated[t].squeeze()
        next_value_t = next_value if t == T - 1 else values[t + 1].squeeze()
        delta = rewards[t].squeeze() + gamma * next_value_t * next_non_terminal - values[t].squeeze()
        advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae
        last_gae = advantages[t]

    returns = advantages + values
    return returns, advantages

def layer_init(layer, std=None, bias_const=0.0):
    """
    Initializes the weights and biases of a given layer.
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    if std is None:
        std = np.sqrt(2)
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
        log_std = torch.clamp(self.log_std, -2, 2) 
        std = log_std.exp().expand_as(mean)
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

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)
        
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
        raw_action = dist.sample()  
        action = torch.tanh(raw_action)  
        action_log_prob = dist.log_prob(raw_action).sum(dim=-1)

        log_prob = action_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy()
    
    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_updates = 0

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
                entropy_loss = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_loss

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

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_updates += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return total_actor_loss / num_updates, total_critic_loss / num_updates
    
    def train(self, cfg, save_interval=10, log_interval=1):
        """ Train the PPO agent """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models/{timestamp}_seed{cfg.env.seed}"
        os.makedirs(save_dir, exist_ok=True)

        global_step = 0
        episode_rewards = []

        for episode in range(cfg.train.num_episodes):
            state, _ = self.envs.reset()
            done = np.zeros(cfg.env.num_envs, dtype=bool)
            episode_reward = np.zeros(cfg.env.num_envs)
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

            while not np.any(done):
                action, action_log_prob = self.get_action(state)
                with torch.no_grad():
                    value = self.critic(torch.FloatTensor(state).to(device)).cpu().numpy().squeeze()

                next_state, reward, terminations, truncations, _ = self.envs.step(action)
                done = np.logical_or(terminations, truncations)

                states.append(state)
                actions.append(action)
                log_probs.append(action_log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                state = next_state
                episode_reward += reward
                global_step += 1

            with torch.no_grad():
                next_value = self.critic(torch.FloatTensor(next_state).to(device)).cpu().numpy().squeeze()

            returns, advantages = compute_gae(np.array(rewards), np.array(values), next_value, np.array(dones), self.gamma, self.gae_lambda)
            actor_loss, critic_loss = self.update(np.array(states), np.array(actions), np.array(log_probs), returns, advantages)
            
            episode_rewards.append(episode_reward.mean())
            wandb.log({
                "episode": episode + 1,
                "average_reward": episode_reward.mean(),
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "global_step": global_step
            })

            if (episode + 1) % save_interval == 0:
                last_actor_path = f"{save_dir}/actor_ep{episode+1}.pth"
                torch.save(self.actor.state_dict(), last_actor_path)
                log.info(f"Saved model at Episode {episode + 1}: {last_actor_path}")

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                log.info(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}, Actor Loss: {actor_loss:.4f}")
        
        return last_actor_path 

    def evaluate(self, env_name, actor_path, num_episodes=10, render=False, save_video_path="videos"):

        os.makedirs(save_video_path, exist_ok=True)
        env = gym.make(env_name, render_mode="rgb_array" if not render else "human")
        env = RecordVideo(env, video_folder=save_video_path, episode_trigger=lambda x: True)

        self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        self.actor.eval()

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    if len(state.shape) == 1:
                        state = np.expand_dims(state, axis=0)  
                    state_tensor = torch.FloatTensor(state).to(device)
                    mean, _ = self.actor(state_tensor)  
                    # action = mean.cpu().numpy()[0]  
                    action = torch.tanh(mean).cpu().numpy()[0]

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            wandb.log({
                "evaluation_episode": episode + 1,
                "evaluation_reward": total_reward
            })

            log.info(f"Episode {episode+1}: Reward: {total_reward:.2f}")
        env.close()
        log.info(f"Videos saved to: {save_video_path}")

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project="RL_PPO", config=dict(cfg))
    envs = gym.make_vec(cfg.env.name, num_envs=cfg.env.num_envs, render_mode="human" if cfg.train.render else None)
    agent = PPO(envs, cfg)
    
    # model_path = agent.train(cfg, save_interval=cfg.train.save_interval, log_interval=cfg.train.log_interval)
    agent.evaluate(cfg.env.name, actor_path="/home/zhikai/code/RL_study/0.05.pth", num_episodes=cfg.eval.eval_episodes, render=cfg.eval.render, save_video_path="videos")
    wandb.finish()

if __name__ == "__main__":
    main()