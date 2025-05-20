# How to run:
# python example1_simple.py --env_name CartPole-v1

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
#import gym
import gymnasium as gym
#from gym.spaces import Discrete, Box
from gymnasium.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    # Print type of action space
    print("Action space type:", "discrete" if isinstance(env.action_space, gym.spaces.Discrete) else "other")
    # Print number of actions
    print("Number of actions:", env.action_space.n)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # *************************************************************************
    # ************** Making the Policy Network *******************************
    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()
    # *************************************************************************
    # *************************************************************************
    
    # *************************************************************************
    # *************** Making the Loss Function *******************************
    # # L(\theta) = -E [ log pi_theta(a_t | s_t) * R(\tau) ]
    # make loss function whose gradient, for the right data, is policy gradient
    # The right data means a set of (state, action, weight) tuples collected while acting according to the current policy, 
    # where the weight for a state-action pair is the return from the episode to which it belongs.
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    # *************************************************************************
    # *************************************************************************

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # ********************************************************************
    # ******************* Running One Epoch of Training *******************
    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        #obs = env.reset()       # first obs comes from starting distribution
        obs, _ = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            #obs, rew, done, _ = env.step(act)
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                #obs, done, ep_rews = env.reset(), False, []
                obs, _ = env.reset()
                done = False
                ep_rews = []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        #batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
        #                          act=torch.as_tensor(batch_acts, dtype=torch.int32),
        #                          weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        #                          )
        batch_loss = compute_loss(obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
                          act=torch.as_tensor(batch_acts, dtype=torch.int32),
                          weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens
    # ********************************************************************
    # ********************************************************************
    
    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        
    torch.save(logits_net.state_dict(), 'trained_policy.pt')
    print("Saved trained policy to 'trained_policy.pt'")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1') 
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)