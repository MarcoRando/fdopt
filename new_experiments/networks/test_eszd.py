import torch

import gym
from gym import wrappers

from eszd import ESZD
from eszd.directions import RandomCoords

from three_layer import ThreeLayerNetwork

in_features = 8
out_features = 4

dtype = torch.float32
device='cuda'

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
    
policy = ThreeLayerNetwork(in_features = in_features, out_features = out_features, dtype=dtype, device=device)

#P = RandomCoords(d = d, l = l, dtype = dtype)

x = torch.ones((1, in_features), device=device)


for i in range(len(policy.model)):
    if isinstance(policy.model[i], torch.nn.Linear):
        print(policy.model[i].weight.shape)

#def target(w, z_sampler = None):
#    policy.model


