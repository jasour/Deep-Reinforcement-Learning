import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym

from gymnasium.spaces import Box, Discrete

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def run_trained_policy(env_name='CartPole-v1', model_path='trained_policy.pt', episodes=5):
    env = gym.make(env_name, render_mode="human")

    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Build same network architecture as before
    logits_net = mlp([obs_dim, 32, n_acts])
    logits_net.load_state_dict(torch.load(model_path))
    logits_net.eval()

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_rew = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = get_policy(obs_tensor).sample().item()
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rew += rew
        print(f"Episode {ep + 1} return: {total_rew}")
    env.close()

if __name__ == '__main__':
    run_trained_policy()