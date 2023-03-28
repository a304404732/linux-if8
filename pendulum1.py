import gym
import torch
import numpy as np
from tqdm import tqdm
from Config import Config
from Agent import Agent

args = Config()
env = gym.make('Pendulum-v1')
np.random.seed(0)
env.seed(0)
agent = Agent(args)
tr = -10.0
loss_list, return_list = [], []
for epi in tqdm(range(args.episode)):
    # 重置环境
    state = env.reset()
    # 使用s,a,r时只知道当前的状态， action和reward 都作为padding
    states = torch.from_numpy(state).reshape(1, args.state_dim).to(device=args.device, dtype=torch.float32)
    actions = torch.zeros((0, args.action_dim), device=args.device, dtype=torch.float32)
    rewards = torch.zeros(0, device=args.device, dtype=torch.float32)
    ep_return = tr
    target_return = torch.tensor(ep_return, device=args.device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=args.device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0
    for t in range(args.max_step):
        actions = torch.cat([actions, torch.zeros((1, args.action_dim), device=args.device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=args.device)])
        # action = env.action_space.sample()
        # 智能体选择动作
        action = agent.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        # 保存一次动作
        actions[-1] = action
        action = action.detach().cpu().numpy()
        action = action * 2
        state_, reward, done, info = env.step(action)
        agent.memory.store_sub_memory(state, action, reward)
        cur_state = torch.from_numpy(state).to(device=args.device).reshape(1, args.state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=args.device, dtype=torch.long) * (t + 1)], dim=1)
        episode_return += reward
        episode_length += 1
        # if done:
        #     break
    # print(episode_return)
    # 将一回合的经验添加到总经验池中
    agent.memory.update_memory()
    # 清空子经验池
    agent.memory.clear_sub_memory()
    # 回合结束进行学习
    if agent.memory.mem_is_full():
        loss = agent.learn()
        loss_list.extend(loss)
    return_list.append(episode_return)

np.savetxt('return.txt', return_list, fmt='%.4f')
np.savetxt('loss.txt', loss_list, fmt='%.4f')
