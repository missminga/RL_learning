"""
PPO (Proximal Policy Optimization, 近端策略优化) 核心算法模块

=== 从 A2C 到 PPO ===

上一个算法 A2C 的更新是这样的：跑完一个回合，用 GAE 算出每步的优势，
然后**做一次梯度下降**更新 Actor 和 Critic。

A2C 有两个"浪费"和"危险"的地方：
1. **数据只用一次就扔**：辛辛苦苦采样来的一整回合数据，只更新一步就丢掉，
   采样效率低。
2. **一步可能迈太大**：策略梯度方向只在"当前策略附近"才可靠。如果学习率/优势
   偏大，一次更新就可能把策略推到很远的地方，新策略在那里的行为完全没被数据
   验证过，结果就是表现突然崩塌（A2C 训练后期的"暴跌"就是这么来的，我们靠
   学习率衰减来缓解，但那只是治标）。

PPO 用一个非常聪明的办法同时解决这两点：**限制每次更新后新旧策略的差距**，
在这个"安全范围"内，就可以放心地把同一批数据**反复用很多次**（多个 epoch、
分 minibatch），既省数据又不会迈过头。

=== 核心思想：重要性采样比值 + 裁剪 ===

PPO 的策略损失基于"重要性采样比值"：
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            = 新策略选这个动作的概率 / 采样时旧策略选它的概率

- r > 1：新策略比旧策略更想选这个动作
- r < 1：新策略比旧策略更不想选这个动作
- r = 1：新旧一致（第一个 epoch 还没更新时就是 1）

朴素的目标是最大化 r_t · A_t（优势大的动作，提高它的概率）。但如果不加限制，
优化器会为了让某个高优势动作的概率涨上天而把 r 推到很大——这就是"迈太大"。

PPO 的**裁剪 (clip)** 目标把步子焊死在一个小区间里：

    L = min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t )

   其中 ε 通常取 0.2，意思是"新旧策略对某个动作的概率比，最多变化 ±20%"。

为什么是 min + clip 这个组合？直觉这样理解：
- A > 0（好动作，想提高概率）：r 一旦涨过 1+ε，clip 把它摁回 1+ε，
  目标不再随 r 增大而变好 → 优化器没动力继续推大 r → 步子被限制住。
- A < 0（坏动作，想降低概率）：r 一旦跌破 1-ε，同理被摁住。
- 外面那层 min 保证裁剪只会让目标"更保守"，不会因为裁剪反而变得更激进
  （这是 PPO 论文里 min 的精髓：取悲观的那个，防止占便宜）。

一句话：**PPO = A2C 的优势加权，但把"每步能改多少"用 clip 锁死，
于是可以安全地把一批数据反复用很多遍。**

=== PPO 一次更新做什么 ===

1. 用当前（旧）策略跑一个回合，记录 (state, action, old_log_prob, value, reward)。
2. 用 GAE 算优势 A_t 和回报目标 returns（这部分和 A2C 一模一样）。
3. 把优势标准化（减均值除标准差）——PPO 的常规操作，让不同回合的优势量级一致、
   裁剪阈值 ε 才有稳定意义。
4. **重复 K 个 epoch**，每个 epoch 把数据打乱、切成若干 minibatch，对每个 minibatch：
       重新前向算出 new_log_prob、value、entropy
       ratio = exp(new_log_prob - old_log_prob)
       actor_loss  = -min(ratio·A, clip(ratio,1-ε,1+ε)·A).mean()   ← 裁剪的策略损失
       critic_loss =  (value - returns)²                            ← 和 A2C 一样
       entropy_bonus 鼓励探索
       反向传播、更新
   注意 old_log_prob 是采样时就固定下来的常数；new_log_prob 随着更新不断变化，
   所以从第 2 个 minibatch 起 ratio 就不再是 1 了，clip 开始真正起作用。

=== A2C vs PPO 对比 ===

| 特性          | A2C                      | PPO                            |
|--------------|--------------------------|--------------------------------|
| 数据复用      | 一批只更新 1 次            | 一批更新 K 个 epoch（反复用）   |
| 限制步长      | 无（靠小学习率/衰减硬撑）  | clip 把新旧策略比锁在 [1-ε,1+ε] |
| 采样效率      | 低                        | 高                             |
| 稳定性        | 后期易崩溃                 | 裁剪保证不迈过头，稳得多         |
| 损失          | actor + critic + 熵       | 裁剪 actor + critic + 熵        |

PPO 是目前工业界和研究界**最主流**的策略优化算法：稳定、好调、效果好。
ChatGPT 等大模型的 RLHF（基于人类反馈的强化学习）就是用 PPO 微调的。
理解了 A2C，PPO 只是在它之上加了"裁剪 + 数据复用"两板斧。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 直接复用 A2C 的网络结构：PPO 和 A2C 用的是**完全相同**的演员-评委网络，
# 区别只在"怎么用这批数据更新它"，不在网络本身。复用它能让两者的差异更清楚。
from core.actor_critic import ActorCriticNetwork
from core.random_utils import seed_env, set_global_seed


class PPOAgent:
    """
    PPO 智能体（Proximal Policy Optimization）

    和 A2C 智能体对比，多记录/多了几样东西：
    - 多存了每步的 state 和 action（因为更新时要"重新前向"算新策略下的 log_prob，
      A2C 一次性更新不需要重存）
    - old_log_probs：采样时旧策略的 log 概率，作为 ratio 的分母（常数）
    - 新增超参：clip_eps（裁剪范围）、update_epochs（一批数据复用几轮）、
      minibatch_size（每个小批多大）

    训练流程仍是回合制（和 A2C 一致），区别全在 update()：
    A2C 是"算一次梯度走一步"，PPO 是"在裁剪保护下，把这批数据反复用 K 轮"。
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=3e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        minibatch_size=64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda  # GAE 的 λ，权衡优势的偏差与方差
        self.clip_eps = clip_eps  # 裁剪范围 ε：新旧策略概率比被锁在 [1-ε, 1+ε]
        self.value_coef = value_coef  # Critic 损失权重 c1
        self.entropy_coef = entropy_coef  # 熵奖励权重 c2
        self.update_epochs = update_epochs  # 同一批数据反复训练几个 epoch（PPO 的精髓）
        self.minibatch_size = minibatch_size  # 每个 minibatch 的样本数
        self.lr0 = lr  # 初始学习率（训练中线性衰减，见 anneal_lr）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # 一个回合内的记录（采样阶段填充，update 时消费）
        self.states = []  # 每步的状态（更新时要重新前向用）
        self.actions = []  # 每步采的动作
        self.log_probs = []  # 旧策略下该动作的 log 概率（常数，ratio 的分母）
        self.values = []  # 采样时 Critic 估的 V(s)（用来算 GAE）
        self.rewards = []  # 每步奖励

    def select_action(self, state):
        """
        按当前（旧）策略采样动作，并把更新时需要的量都记录下来

        和 A2C 的关键区别：这里把 log_prob 和 value **detach** 成常数存起来
        （它们代表"采样时那一刻的旧策略"，更新时不能再对它们求导），
        而且额外存了 state 和 action —— 因为 PPO 更新时要拿这些状态
        重新跑一遍网络，得到"新策略"下的 log_prob 来算比值 ratio。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():  # 采样阶段不需要梯度
            logits, value = self.net(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action.item()))
        self.log_probs.append(log_prob.squeeze())  # 旧 log 概率（常数）
        self.values.append(value.squeeze())  # 旧 V(s)（常数）

        return action.item()

    def store_reward(self, reward):
        """记录一步奖励"""
        self.rewards.append(reward)

    def anneal_lr(self, frac):
        """
        按训练进度线性把学习率从 lr0 降到 lr0 的 1/10。

        PPO 的裁剪已经能很好地防止"迈太大"，所以它本身就比 A2C 稳得多；
        学习率衰减在这里属于"锦上添花"，让后期收敛更平滑、最终策略更稳。

        参数:
            frac: 训练进度，0（开始）→ 1（结束）
        """
        frac = min(max(frac, 0.0), 1.0)
        lr = self.lr0 * (1.0 - 0.9 * frac)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, last_state=None, bootstrap=False):
        """
        回合结束后，用裁剪目标把这批数据**反复训练 K 个 epoch**

        参数:
            last_state: 回合最后到达的状态（仅 bootstrap 时用）
            bootstrap:  回合是否因"截断(超过最大步数)"结束。截断时杆还没倒，
                        用 V(末状态) 自举估计后续价值（和 A2C 完全一致）。

        返回: (total_loss, actor_loss, critic_loss)，是所有 minibatch 更新的平均
        """
        if not self.rewards:
            return 0.0, 0.0, 0.0

        device = self.device
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        old_log_probs = torch.stack(self.log_probs).to(device)  # [T]，常数
        values = torch.stack(self.values).to(device)  # [T]，常数
        rewards_t = torch.FloatTensor(self.rewards).to(device)

        # ===== 第一步：用 GAE 算优势 A_t 和回报目标 returns（和 A2C 完全相同）=====
        # δ_t = r_t + γ·V(s_{t+1}) − V(s_t)；A_t = δ_t + (γλ)·δ_{t+1} + ...
        # 末状态价值：真正失败结束(terminated)→0；被截断(truncated)→用 V(末状态) 自举
        if bootstrap and last_state is not None:
            with torch.no_grad():
                s = torch.FloatTensor(last_state).unsqueeze(0).to(device)
                _, last_v = self.net(s)
                last_v = last_v.squeeze().reshape(1)
        else:
            last_v = torch.zeros(1, device=device)
        next_values = torch.cat([values[1:], last_v])

        deltas = rewards_t + self.gamma * next_values - values
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for i in range(len(deltas) - 1, -1, -1):
            gae = deltas[i] + self.gamma * self.gae_lambda * gae
            advantages[i] = gae
        returns = advantages + values  # Critic 的回归目标

        # ===== 第二步：标准化优势（PPO 的常规操作）=====
        # 把优势减均值除标准差，让它均值约 0、尺度约 1。这样裁剪阈值 ε 才有
        # 稳定一致的意义（否则不同回合优势量级忽大忽小，同一个 ε 时松时紧）。
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===== 第三步：把这批数据反复训练 K 个 epoch，每个 epoch 分 minibatch =====
        n = len(rewards_t)
        indices = np.arange(n)
        sum_total, sum_actor, sum_critic, n_updates = 0.0, 0.0, 0.0, 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)  # 每个 epoch 重新打乱，去相关、更稳
            for start in range(0, n, self.minibatch_size):
                mb = indices[start : start + self.minibatch_size]
                mb_idx = torch.LongTensor(mb).to(device)

                # 用"当前（新）策略"重新前向：这是 PPO 和 A2C 最本质的不同——
                # A2C 只在采样时前向一次；PPO 每个 minibatch 都重新前向，
                # 这样才能拿到随更新不断变化的 new_log_prob 来算比值 ratio。
                logits, value_pred = self.net(states[mb_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                mb_adv = advantages[mb_idx]

                # 重要性采样比值 ratio = π_new / π_old = exp(log_new - log_old)
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])

                # 裁剪的策略损失：取"未裁剪"和"裁剪"两者中更悲观（更小）的那个
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic 损失：让 V(s) 回归到 GAE 目标 returns（和 A2C 一样）
                critic_loss = nn.functional.mse_loss(
                    value_pred.squeeze(-1), returns[mb_idx]
                )

                # 熵奖励：鼓励保持随机性（最大化熵 = 最小化 -熵）
                entropy_loss = -entropy

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                sum_total += float(loss.item())
                sum_actor += float(actor_loss.item())
                sum_critic += float(critic_loss.item())
                n_updates += 1

        # 清空回合记录
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

        n_updates = max(n_updates, 1)
        return sum_total / n_updates, sum_actor / n_updates, sum_critic / n_updates


def train_ppo(
    env, agent, episodes=500, max_steps=500, on_episode_end=None, should_stop=None
):
    """
    训练 PPO 智能体

    训练骨架和 train_a2c 几乎逐行一致（同为回合制），唯一的不同在 agent.update()
    内部——A2C 只更新一步，PPO 在裁剪保护下把这批数据反复用 K 个 epoch。

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
        # 随训练进度衰减学习率（PPO 本身已很稳，这里是锦上添花）
        agent.anneal_lr(ep / max(episodes, 1))
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

        # 回合结束更新。若被截断（杆没倒只是到步数上限），用末状态价值自举
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


def run_ppo_experiment(
    episodes=600,
    hidden_dim=128,
    lr=3e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    update_epochs=10,
    minibatch_size=64,
    max_steps=500,
    n_runs=1,
    seed=None,
    on_episode_end=None,
    should_stop=None,
):
    """
    运行 PPO CartPole 实验，返回 JSON 可序列化的 dict

    供 Web API 调用。返回结构与 A2C 实验完全一致（含 actor / critic 两条损失曲线），
    方便在前端直接对比"A2C 和 PPO 的学习曲线有何不同"。
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

        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        ep_rewards, ep_steps, ep_losses, ep_actor, ep_critic = train_ppo(
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
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "update_epochs": update_epochs,
            "minibatch_size": minibatch_size,
            "n_runs": n_runs,
        },
    }
