import torch

import gym
from gym import wrappers

from eszd import ESZD
from eszd.directions import RandomCoords

from three_layer import ThreeLayerNetwork

d = 8
l = 8

dtype = torch.float32

def eval_policy(policy, env, n_steps=200):
    
    total_reward = 0
    obs, _ = env.reset()
    for i in range(n_steps):
        env.render()
        action = policy.predict(torch.tensor(obs).reshape(1, -1), scale="softmax")
        new_obs, reward, done, _, _ = env.step(action.item())        
        total_reward = total_reward + reward
        obs = new_obs
        if done:
            break

    return total_reward
    


policy = ThreeLayerNetwork(d = d, l = l)

P = RandomCoords(d = d, l = l, dtype = dtype)
