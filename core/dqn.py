"""
DQN (Deep Q-Network) 核心算法模块

从 Q 表到神经网络：
- Q-Learning 用一张表 Q(s,a) 记录每个状态-动作对的价值
- 但当状态空间很大或是连续的（比如 CartPole 的4个浮点数），表格装不下
- DQN 的思路：用神经网络来"逼近" Q 函数，输入状态，输出各动作的 Q 值

DQN 的三个关键技巧：
1. 经验回放 (Experience Replay): 把经历存起来，随机抽样学习，打破时间相关性
2. 目标网络 (Target Network): 用一个"慢更新"的网络计算目标值，稳定训练
3. ε-贪心探索: 和 Q-Learning 一样，平衡探索与利用
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Q 网络：输入状态 → 输出每个动作的 Q 值

    结构非常简单：两层全连接 + ReLU 激活
    输入维度 = 状态维度（CartPole 是 4）
    输出维度 = 动作数量（CartPole 是 2：左推 / 右推）
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    经验回放缓冲区

    为什么需要经验回放？
    - 连续收集的经验有很强的时间相关性（前后帧很像）
    - 直接用来训练神经网络会导致不稳定
    - 把经验存起来，随机抽一批来学习 → 打破相关性，更稳定

    存的每条经验是一个五元组：(state, action, reward, next_state, done)
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机抽一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN 智能体

    包含两个网络：
    - policy_net（策略网络）：用来选动作和训练更新
    - target_net（目标网络）：用来计算 TD 目标值，定期从 policy_net 复制参数

    为什么要两个网络？
    想象你在考试时，答案和评分标准同时在变——你很难进步。
    目标网络就是一个"固定的评分标准"，让训练更稳定。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64,
                 target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 选择设备（有 GPU 用 GPU，没有用 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络（主网络，用来选动作和学习）
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # 目标网络（慢更新，用来计算目标 Q 值）
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # 初始时两个网络参数相同
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 记录训练了多少回合，用于决定何时更新目标网络
        self.episodes_done = 0

    def select_action(self, state):
        """
        ε-贪心选动作

        - 以 ε 概率随机选（探索）
        - 以 1-ε 概率选 Q 值最大的（利用）
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """把一步经验存入回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        从经验回放中抽样，执行一步梯度更新

        核心公式（和 Q-Learning 一样，只是用神经网络来算）：
        目标: y = r + γ * max_a' Q_target(s', a')    （如果 done 则 y = r）
        损失: L = (Q_policy(s, a) - y)²
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 从缓冲区随机抽一批
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # 转为 PyTorch 张量
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值：Q_policy(s, a)
        # gather 的作用：从所有动作的 Q 值中，挑出实际执行的那个动作的 Q 值
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 目标 Q 值：r + γ * max Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # 计算损失并反向传播
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """把策略网络的参数复制到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.episodes_done += 1


def train_dqn(env, agent, episodes=300, max_steps=500,
              on_episode_end=None):
    """
    训练 DQN 智能体

    参数:
        env: Gymnasium 环境
        agent: DQNAgent 实例
        episodes: 训练回合数
        max_steps: 每回合最大步数
        on_episode_end: 回调函数，每回合结束时调用

    返回:
        episode_rewards: 每回合的总奖励
        episode_steps: 每回合的步数
        losses: 每回合的平均损失
    """
    episode_rewards = []
    episode_steps = []
    losses = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        ep_losses = []

        for step in range(max_steps):
            # 选动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存经验
            agent.store_transition(state, action, reward, next_state, done)

            # 训练一步
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        # 回合结束：衰减 ε，定期更新目标网络
        agent.decay_epsilon()
        if (ep + 1) % agent.target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        losses.append(avg_loss)

        if on_episode_end:
            on_episode_end(ep, total_reward, step + 1, agent.epsilon)

    return episode_rewards, episode_steps, losses


def run_cartpole_experiment(episodes=300, hidden_dim=128, lr=1e-3,
                            gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                            epsilon_min=0.01, buffer_capacity=10000,
                            batch_size=64, target_update_freq=10,
                            max_steps=500, n_runs=1):
    """
    运行 CartPole DQN 实验，返回 JSON 可序列化的 dict

    供 Web API 调用。
    """
    import gymnasium as gym

    all_rewards = np.zeros(episodes)
    all_steps = np.zeros(episodes)
    all_losses = np.zeros(episodes)

    for _ in range(n_runs):
        env = gym.make("CartPole-v1")

        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dim=hidden_dim,
            lr=lr, gamma=gamma,
            epsilon=epsilon, epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
        )

        ep_rewards, ep_steps, ep_losses = train_dqn(
            env, agent, episodes=episodes, max_steps=max_steps,
        )

        all_rewards += np.array(ep_rewards)
        all_steps += np.array(ep_steps)
        all_losses += np.array(ep_losses)

        env.close()

    avg_rewards = (all_rewards / n_runs).tolist()
    avg_steps = (all_steps / n_runs).tolist()
    avg_losses = (all_losses / n_runs).tolist()

    # 判断是否"解决"了 CartPole（连续 100 回合平均 >= 475）
    solved_episode = None
    if len(avg_rewards) >= 100:
        for i in range(99, len(avg_rewards)):
            window_avg = np.mean(avg_rewards[max(0, i - 99):i + 1])
            if window_avg >= 475:
                solved_episode = i - 99
                break

    return {
        "episodes": episodes,
        "avg_rewards": avg_rewards,
        "avg_steps": avg_steps,
        "avg_losses": avg_losses,
        "summary": {
            "final_avg_reward": round(float(np.mean(avg_rewards[-50:])), 1),
            "final_avg_steps": round(float(np.mean(avg_steps[-50:])), 1),
            "max_reward": round(float(np.max(avg_rewards)), 1),
            "solved_episode": solved_episode,
        },
    }
