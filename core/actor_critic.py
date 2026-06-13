"""
Actor-Critic (A2C, Advantage Actor-Critic) 核心算法模块

=== 从 REINFORCE 到 A2C ===

上一个算法 REINFORCE（策略梯度）的更新公式是：
    θ ← θ + α * G_t * ∇θ log π(a_t|s_t)
其中 G_t 是从 t 时刻往后的折扣回报。它有个大毛病——**方差太大**：
G_t 是把一整条轨迹的随机奖励加起来，运气好坏会让它剧烈波动，
导致训练很不稳定、收敛慢。

REINFORCE 里我们用了"回报标准化"（减均值除标准差）来缓解，
但那只是个粗糙的技巧。A2C 给出了更优雅的办法：**引入一个 Critic（评委）**。

=== A2C 的核心思想：演员 + 评委 ===

A2C 同时训练两个东西：
- **Actor（演员）**：策略网络 π(a|s)，决定"在状态 s 下选哪个动作"。
  —— 这部分和 REINFORCE 一模一样。
- **Critic（评委）**：价值网络 V(s)，估计"状态 s 大概能拿到多少长期回报"。
  —— 这部分像 DQN 学的价值，但 Critic 学的是"状态价值 V(s)"，
     不是"动作价值 Q(s,a)"。

关键在于一个叫"优势 (Advantage)"的量：
    A(s, a) = G_t - V(s)
                └─ Critic 给出的"基准线 (baseline)"

含义："这个动作实际拿到的回报 G_t，比这个状态'本来平均能拿到的'(V(s)) 好多少？"
- A > 0：这个动作比平均水平好 → 增大它的概率
- A < 0：这个动作比平均水平差 → 减小它的概率

=== 为什么用优势代替回报能降低方差？===

REINFORCE 直接用 G_t（绝对回报）。比如 CartPole 里 G_t 总是正的（每步 +1），
于是所有动作都被"鼓励"，只是程度不同——信号很模糊。
A2C 用 A = G_t - V(s)（相对回报）：减掉了"这个状态本来就值多少"，
只保留"这个动作带来的额外好处"，信号更干净，方差更小，训练更稳。

而且：减去 V(s) 这个 baseline **不会改变梯度的期望**（数学上可证），
所以是"免费的午餐"——只降方差，不引入偏差。

=== 两个损失函数 ===

A2C 一次更新同时优化两个网络：
1. Actor 损失（策略梯度，和 REINFORCE 同形，只是把 G_t 换成 advantage）：
       actor_loss = -Σ log π(a_t|s_t) * A_t      （A_t 视为常数，不回传梯度）
2. Critic 损失（让 V(s) 逼近真实回报 G_t，就是个回归问题）：
       critic_loss = Σ (V(s_t) - G_t)²
3. （可选）熵奖励：鼓励策略保持一定随机性，防止过早收敛到次优：
       entropy_bonus = Σ H(π(·|s_t))

   总损失 = actor_loss + c1 * critic_loss - c2 * entropy

=== 三种方法对比 ===

| 特性       | DQN              | REINFORCE        | A2C                    |
|-----------|------------------|------------------|------------------------|
| 学什么     | Q(s,a) 价值       | π(a|s) 策略       | π(a|s) 策略 + V(s) 价值 |
| 谁选动作   | argmax Q（评委）  | 策略采样（演员）   | 策略采样（演员）        |
| 评估信号   | TD 目标           | 蒙特卡洛回报 G_t   | 优势 A = G_t - V(s)    |
| 方差       | 低（但有偏）       | 高               | 中（兼顾偏差与方差）     |
| 网络数量   | 2（策略+目标）     | 1（策略）         | 2（演员网络+评委网络）  |

A2C 是现代算法（PPO、A3C、SAC 等）的基础，理解它就打通了
"基于价值"和"基于策略"两条路线的结合点。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.random_utils import seed_env, set_global_seed


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络：两套**相互独立**的网络

    - Actor（演员）网络：状态 → 各动作的 logits（经 softmax 变概率）—— 决定怎么动
    - Critic（评委）网络：状态 → 一个标量 V(s) —— 评估这个状态值多少

    为什么不共享网络？
    A2C 一次更新里同时算 actor 和 critic 两种损失。如果两者共享底层网络，
    Critic 的损失（回报的平方，量级可达上千）产生的梯度会顺着共享层
    "压制"住 Actor 那份小得多的梯度，结果网络几乎只在拟合价值、策略学不动。
    （这一点在 CartPole 上实测非常明显：共享网络基本学不起来。）
    用两套独立网络，各自的梯度互不干扰，训练稳定可靠。

    激活函数用 Tanh 而非 ReLU：策略梯度类方法对激活函数较敏感，
    Tanh 输出有界、梯度更平滑，在 CartPole 这类任务上收敛更稳更快。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Actor 网络：状态 → 动作 logits
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Critic 网络：状态 → 状态价值 V(s)（一个标量）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """同时返回动作 logits 和 状态价值 V(s)"""
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_probs(self, state):
        """只取动作概率分布（softmax 后）"""
        return torch.softmax(self.actor(state), dim=-1)


class A2CAgent:
    """
    A2C 智能体（Advantage Actor-Critic）

    和 REINFORCE 智能体对比：
    - 多了一个 Critic（价值网络），更新时会用 V(s) 作为 baseline
    - 多记录了每步的 value 和 entropy
    - 损失由 actor_loss + critic_loss(- entropy) 三部分组成

    训练流程（仍然是回合制，和 REINFORCE 一致）：
    1. 跑完一整个回合，记录 (log_prob, value, reward, entropy)
    2. 从后往前算折扣回报 G_t（若回合是被截断而非真正结束，则用 V(末状态) 自举）
    3. 优势 A_t = G_t - V(s_t)
    4. 更新：actor 用 A_t 加权，critic 让 V 逼近 G_t
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=3e-3,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        self.gamma = gamma
        self.value_coef = value_coef  # Critic 损失的权重 c1
        self.entropy_coef = entropy_coef  # 熵奖励的权重 c2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # 一个回合内的记录
        self.log_probs = []  # 每步动作的 log 概率
        self.values = []  # 每步 Critic 估计的 V(s)
        self.rewards = []  # 每步奖励
        self.entropies = []  # 每步策略的熵（衡量随机性）

    def select_action(self, state):
        """
        按策略采样动作，同时让 Critic 评估当前状态

        和 REINFORCE 的区别：这里 forward 一次同时拿到
        动作分布（Actor）和状态价值（Critic）。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.net(state_t)

        dist = Categorical(logits=logits)
        action = dist.sample()

        # 记录更新时需要的量
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value.squeeze())
        self.entropies.append(dist.entropy())

        return action.item()

    def store_reward(self, reward):
        """记录一步奖励"""
        self.rewards.append(reward)

    def update(self, last_state=None, bootstrap=False):
        """
        回合结束后更新 Actor 和 Critic

        参数:
            last_state: 回合最后到达的状态（仅 bootstrap 时用）
            bootstrap:  回合是否因"截断(超过最大步数)"结束。
                        - 真正失败结束(terminated)：末状态价值=0
                        - 被截断(truncated)：杆还没倒，用 V(末状态) 估计后续价值，
                          这样回报估计更准确（n-step 自举的思想）。

        返回: (total_loss, actor_loss, critic_loss)
        """
        if not self.rewards:
            return 0.0, 0.0, 0.0

        # ===== 第一步：计算折扣回报 G_t =====
        # 若回合被截断，用 Critic 对末状态的估值作为"未来回报"的起点（自举）
        if bootstrap and last_state is not None:
            with torch.no_grad():
                state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                _, next_value = self.net(state_t)
                G = next_value.squeeze().item()
        else:
            G = 0.0

        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # ===== 第二步：计算优势 A_t = G_t - V(s_t) =====
        # 注意 .detach()：算 actor 损失时，优势只当"权重"，
        # 不能让梯度从这里回传到 Critic（Critic 由 critic_loss 单独训练）。
        # 这里**不**再对优势做标准化：Critic 本身已经充当了 baseline，
        # 若再按单个回合标准化优势，会把"这一整个回合是好是坏"的信息抹掉
        # （短回合里只剩"步在回合中的先后"这种无用信号），反而学不动。
        advantages = returns - values.detach()

        # ===== 第三步：三部分损失 =====
        # 三个损失都用 mean（按步求平均）而非 sum，这样它们的量级互相可比、
        # 训练更稳定（sum 会让长回合的损失数值忽大忽小）。
        # 1) Actor：让优势大的动作概率上升
        actor_loss = -(log_probs * advantages).mean()
        # 2) Critic：让 V(s) 回归到真实回报 G_t（回归问题）
        critic_loss = nn.functional.mse_loss(values, returns)
        # 3) 熵奖励：鼓励保持随机性（最大化熵 = 最小化 -熵）
        entropy_loss = -entropies.mean()

        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            + self.entropy_coef * entropy_loss
        )

        # ===== 第四步：反向传播，一次更新整个网络 =====
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        a_loss = float(actor_loss.item())
        c_loss = float(critic_loss.item())
        t_loss = float(total_loss.item())

        # 清空回合记录
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

        return t_loss, a_loss, c_loss


def train_a2c(
    env, agent, episodes=500, max_steps=500, on_episode_end=None, should_stop=None
):
    """
    训练 A2C 智能体

    和 REINFORCE 训练几乎一样（同为回合制更新），区别在于：
    - update() 时会区分"真正结束"和"被截断"，截断时用 Critic 自举
    - 损失拆成 actor / critic 两部分分别记录

    返回:
        episode_rewards, episode_steps, total_losses, actor_losses, critic_losses
    """
    episode_rewards = []
    episode_steps = []
    total_losses = []
    actor_losses = []
    critic_losses = []

    for ep in range(episodes):
        if should_stop and should_stop():
            break
        state, _ = env.reset()
        total_reward = 0.0
        truncated = False
        next_state = state

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        # 回合结束更新。若是被截断（杆没倒只是到步数上限），用末状态价值自举
        t_loss, a_loss, c_loss = agent.update(
            last_state=next_state, bootstrap=truncated
        )

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        total_losses.append(t_loss)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        if on_episode_end:
            on_episode_end(ep, total_reward, step + 1, t_loss)

    return episode_rewards, episode_steps, total_losses, actor_losses, critic_losses


def run_a2c_experiment(
    episodes=500,
    hidden_dim=128,
    lr=3e-3,
    gamma=0.99,
    value_coef=0.5,
    entropy_coef=0.01,
    max_steps=500,
    n_runs=1,
    seed=None,
    on_episode_end=None,
    should_stop=None,
):
    """
    运行 A2C CartPole 实验，返回 JSON 可序列化的 dict

    供 Web API 调用。结构与 REINFORCE 实验一致，额外多返回
    actor / critic 两条损失曲线，方便直观看到"评委"是怎么学的。
    """
    import gymnasium as gym

    set_global_seed(seed)
    all_rewards = np.zeros(episodes)
    all_steps = np.zeros(episodes)
    all_losses = np.zeros(episodes)
    all_actor_losses = np.zeros(episodes)
    all_critic_losses = np.zeros(episodes)

    for run_idx in range(n_runs):
        if should_stop and should_stop():
            break
        env = gym.make("CartPole-v1")
        seed_env(env, None if seed is None else seed + run_idx)

        agent = A2CAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

        ep_rewards, ep_steps, ep_losses, ep_actor, ep_critic = train_a2c(
            env,
            agent,
            episodes=episodes,
            max_steps=max_steps,
            should_stop=should_stop,
            on_episode_end=(
                lambda ep, reward, steps, loss: on_episode_end(
                    run_idx, ep, reward, steps, loss
                )
            )
            if on_episode_end
            else None,
        )

        all_rewards[: len(ep_rewards)] += np.array(ep_rewards)
        all_steps[: len(ep_steps)] += np.array(ep_steps)
        all_losses[: len(ep_losses)] += np.array(ep_losses)
        all_actor_losses[: len(ep_actor)] += np.array(ep_actor)
        all_critic_losses[: len(ep_critic)] += np.array(ep_critic)

        env.close()

    avg_rewards = (all_rewards / n_runs).tolist()
    avg_steps = (all_steps / n_runs).tolist()
    avg_losses = (all_losses / n_runs).tolist()
    avg_actor_losses = (all_actor_losses / n_runs).tolist()
    avg_critic_losses = (all_critic_losses / n_runs).tolist()

    # 判断是否"解决"了 CartPole（连续 100 回合平均 >= 475）
    solved_episode = None
    if len(avg_rewards) >= 100:
        for i in range(99, len(avg_rewards)):
            window_avg = np.mean(avg_rewards[max(0, i - 99) : i + 1])
            if window_avg >= 475:
                solved_episode = i - 99
                break

    return {
        "episodes": episodes,
        "avg_rewards": avg_rewards,
        "avg_steps": avg_steps,
        "avg_losses": avg_losses,
        "avg_actor_losses": avg_actor_losses,
        "avg_critic_losses": avg_critic_losses,
        "summary": {
            "final_avg_reward": round(float(np.mean(avg_rewards[-50:])), 1),
            "final_avg_steps": round(float(np.mean(avg_steps[-50:])), 1),
            "max_reward": round(float(np.max(avg_rewards)), 1),
            "solved_episode": solved_episode,
        },
        "seed": seed,
        "params": {
            "episodes": episodes,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "gamma": gamma,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "n_runs": n_runs,
        },
    }
