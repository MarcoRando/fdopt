import gym
import torch
from gym import wrappers
from three_layer import ThreeLayerNetwork
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import uuid

def eval_policy(policy, env, n_steps=200, out="./"):
    
    total_reward = 0
    video = VideoRecorder(env, 'test.mp4')
    obs, _ = env.reset()
    for i in range(n_steps):
        env.render()
        video.capture_frame()
        action = policy.predict(torch.tensor(obs).reshape(1, -1), scale="softmax")
        new_obs, reward, done, _, _ = env.step(action.item())        
        total_reward = total_reward + reward
        obs = new_obs
        if done:
            break
    video.close()

    return total_reward

ENV_INFO = {
    "CartPole-v0": (4, 2),
    "LunarLander-v2": (8, 4),
    "LunarLanderContinuous-v2": (8, 2),
    "MountainCar-v0": (2, 3),
    "MountainCarContinuous-v0": (2, 1),
    "CarRacing-v0": (96*96*3, 3),
    "BipedalWalker-v3": (24, 4)
}


policy = ThreeLayerNetwork(8, 4)

model = "../models/test_LunarLander_v3.pkl"

mname = model.split("/")[-1].split(".")[0]

env = gym.make("LunarLander-v2", render_mode='rgb_array')

os.makedirs(f'videos/{mname}/', exist_ok=True)

#env = VideoRecorder(env, f'videos/{mname}/' + str(uuid.uuid4()))

print(eval_policy(policy, env, n_steps=1600, out=f'videos/{mname}/'))
env.close()