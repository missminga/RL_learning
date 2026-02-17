"""
Policy Gradient (REINFORCE) 核心算法模块

=== 从 Value-Based 到 Policy-Based ===

DQN 的思路：先学 Q(s,a)（每个动作"值多少"），再选 Q 值最大的动作。
REINFORCE 的思路：直接学策略 π(a|s)（在状态 s 下，选各个动作的概率）。

类比：
- DQN 像一个"评委"，给每个动作打分，选分最高的。
- REINFORCE 像一个"演员"，直接学习在什么情况下该做什么。

=== REINFORCE 算法核心 ===

1. 策略网络输出一个概率分布（通过 softmax）：π(a|s) = P(选动作a | 状态s)
2. 按这个概率随机采样一个动作（随机策略，不是贪心）
3. 跑完一整个回合，收集所有 (s, a, r)
4. 计算每一步的回报 G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
5. 更新策略：让回报高的动作概率增大，回报低的动作概率减小

更新公式：θ ← θ + α * G_t * ∇θ log π(a_t|s_t)
等价的损失函数：loss = -log π(a_t|s_t) * G_t

=== 与 DQN 的关键区别 ===

| 特性         | DQN                  | REINFORCE            |
|-------------|----------------------|----------------------|
| 学什么       | Q 值（动作的价值）     | 策略（动作的概率）     |
| 怎么选动作   | ε-贪心（大部分选最优） | 按概率采样（随机策略） |
| 何时更新     | 每一步都可以更新       | 必须跑完整个回合      |
| 需要回放缓冲 | 需要                  | 不需要               |
| 需要目标网络 | 需要                  | 不需要               |
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    策略网络：输入状态 → 输出各动作的概率

    和 DQN 的 Q 网络结构几乎一样，区别只在最后一层：
    - Q 网络：输出 Q 值（任意实数）
    - 策略网络：输出概率（经过 softmax，和为 1）
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
        """输出 logits（未归一化的分数），后续用 softmax 转为概率"""
        return self.net(x)

    def get_action_probs(self, state):
        """输出动作概率分布"""
        logits = self.forward(state)
        return torch.softmax(logits, dim=-1)


class REINFORCEAgent:
    """
    REINFORCE 智能体

    和 DQN 智能体对比：
    - 没有目标网络（不需要稳定目标，因为我们用 Monte Carlo 回报）
    - 没有经验回放（每个回合的经验用完就扔）
    - 选动作是按概率随机的（不是 ε-贪心）

    训练流程：
    1. 跑完一整个回合，记录所有 (log_prob, reward)
    2. 回合结束后，计算每步的折扣回报 G_t
    3. 用 G_t 加权 log_prob 来更新策略
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 一个回合内的记录
        self.log_probs = []   # 每步选择的动作的 log 概率
        self.rewards = []     # 每步获得的奖励

    def select_action(self, state):
        """
        按策略网络输出的概率分布采样动作

        和 DQN 的区别：
        - DQN: ε 概率随机，1-ε 选 Q 值最大的 → 确定性策略 + 强制探索
        - REINFORCE: 始终按概率采样 → 天然有探索性（概率小的也有机会被选到）
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy_net(state_t)

        # Categorical 分布：用 logits 创建离散概率分布
        dist = Categorical(logits=logits)
        action = dist.sample()

        # 记录 log π(a|s)，更新时要用
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        """记录一步的奖励"""
        self.rewards.append(reward)

    def update(self):
        """
        回合结束后更新策略

        核心步骤：
        1. 从后往前计算折扣回报 G_t = r_t + γ*G_{t+1}
        2. 标准化回报（减均值除标准差）—— 减少方差的小技巧
        3. 计算策略梯度损失：loss = -Σ log π(a_t|s_t) * G_t
        4. 反向传播更新网络
        """
        if not self.rewards:
            return 0.0

        # 第一步：计算折扣回报
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # 第二步：标准化回报（baseline 的简单版本）
        # 为什么要标准化？
        # - CartPole 的奖励总是正的（+1/步），所有动作都会被"鼓励"
        # - 减去均值后，好的动作得到正的强化，差的动作得到负的强化
        # - 除以标准差让梯度规模更稳定
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 第三步：计算损失
        # loss = -Σ log π(a_t|s_t) * G_t
        # 为什么加负号？因为 PyTorch 做梯度下降（最小化），我们要最大化期望回报
        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).sum()

        # 第四步：反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_value = loss.item()

        # 清空回合记录，准备下一个回合
        self.log_probs = []
        self.rewards = []

        return loss_value


def train_reinforce(env, agent, episodes=500, max_steps=500,
                    on_episode_end=None):
    """
    训练 REINFORCE 智能体

    和 DQN 训练的区别：
    - DQN: 每一步都训练（从回放缓冲区抽样）
    - REINFORCE: 每回合结束才训练（用整个回合的数据）

    参数:
        env: Gymnasium 环境
        agent: REINFORCEAgent 实例
        episodes: 训练回合数
        max_steps: 每回合最大步数
        on_episode_end: 回调函数

    返回:
        episode_rewards: 每回合总奖励
        episode_steps: 每回合步数
        losses: 每回合的损失值
    """
    episode_rewards = []
    episode_steps = []
    losses = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # 按概率选动作（同时自动记录 log_prob）
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 记录奖励
            agent.store_reward(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        # 回合结束，更新策略（这是和 DQN 最大的区别！）
        loss = agent.update()

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        losses.append(loss)

        if on_episode_end:
            on_episode_end(ep, total_reward, step + 1, loss)

    return episode_rewards, episode_steps, losses


def run_reinforce_experiment(episodes=500, hidden_dim=128, lr=1e-3,
                             gamma=0.99, max_steps=500, n_runs=1):
    """
    运行 REINFORCE CartPole 实验，返回 JSON 可序列化的 dict

    供 Web API 调用。
    """
    import gymnasium as gym

    all_rewards = np.zeros(episodes)
    all_steps = np.zeros(episodes)
    all_losses = np.zeros(episodes)

    for _ in range(n_runs):
        env = gym.make("CartPole-v1")

        agent = REINFORCEAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dim=hidden_dim,
            lr=lr, gamma=gamma,
        )

        ep_rewards, ep_steps, ep_losses = train_reinforce(
            env, agent, episodes=episodes, max_steps=max_steps,
        )

        all_rewards += np.array(ep_rewards)
        all_steps += np.array(ep_steps)
        all_losses += np.array(ep_losses)

        env.close()

    avg_rewards = (all_rewards / n_runs).tolist()
    avg_steps = (all_steps / n_runs).tolist()
    avg_losses = (all_losses / n_runs).tolist()

    # 判断是否"解决"了 CartPole
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
