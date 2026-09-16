"""
连续动作版 PPO（高斯策略）—— 从"选哪个动作"到"每个关节出多大力"

=== 为什么需要这个模块 ===

core/ppo.py 里的 PPO 用 Categorical 分布处理**离散动作**（CartPole：左/右二选一）。
但真实机器人的动作是**连续的**：8 个关节各自输出一个扭矩（一个实数）。
离散分布没法描述"出力 0.35 还是 0.36"，所以要换一个策略分布：

    离散：Categorical(logits)      —— 网络输出每个动作的分数
    连续：Normal(mean, std)        —— 网络输出每个动作维度的均值，标准差单独学

=== 高斯策略（Gaussian Policy）===

策略 π(a|s) 变成一个多元高斯分布（各维度独立）：

    a ~ Normal(μ(s), σ)     μ(s) 是 Actor 网络的输出，σ 是可学习参数

- μ(s)：状态 s 下"最想出的力"，随状态变化（网络算出来）
- σ：探索的"胆子大小"，全局一个（不随状态变），训练中自动越学越小
  —— 初期 σ 大，动作随机，到处探索；后期 σ 小，动作收敛到 μ 附近

log π(a|s) 就是高斯对数概率，各维度独立所以**把各维度的 log 概率相加**：
    log π(a|s) = Σ_i log Normal(a_i | μ_i(s), σ_i)

=== 按步数攒批更新（工业界做法）===

离散版 PPO 是**回合制更新**（跑完一回合更新一次），在 CartPole 上没问题。
但 MuJoCo 机器人任务里回合长度变化剧烈：Ant 训练初期每回合只活 20~30 步
就摔倒，回合制更新意味着每次只拿几十条数据就学一次 —— batch 太小、
梯度噪声太大，根本学不动。

工业界（CleanRL、Isaac Lab 的 rsl_rl）的标准做法是**按步数攒批**：

    跨回合边界连续采样，攒满 batch_size 步（如 2048）才更新一次

这样无论回合长短，每次更新的数据量都恒定。代价是 GAE 要处理
"一批数据里跨了多个回合"的情况，解决方法：

1. 每步记录该步是否回合结束（ends），回合结束处优势不再向后传播
2. terminated（真结束）→ 末状态价值按 0 算
3. truncated（被步数上限截断）→ 用 V(末状态) 自举（采样时就地算好）
4. 攒批停在回合中途 → 用 V(当前状态) 自举

=== 和离散版 PPO 的代码差异 ===

1. 分布换了：Categorical → Normal（log_prob 要对动作维度求和）
2. 动作存储：int → np 数组（[action_dim] 维向量）
3. 攒批方式：回合制 → 按步数攒批（本模块核心改动）
4. 超参不同：学习率更小（3e-4 vs 3e-3），连续控制对步长更敏感

**其余部分和离散版完全一样**：GAE、clip 裁剪、minibatch 复用 K 个 epoch。
这正是 PPO 框架的威力——换分布、换攒批方式都不改算法核心。

=== MuJoCo 环境（真正的"仿真机器人"）===

本模块默认搭配 MuJoCo 物理引擎的环境，比如：
- Ant-v5：四足蚂蚁机器人，8 个关节扭矩（action_dim=8），学会爬行
- HalfCheetah-v5：二维猎豹，6 个关节（action_dim=6），学会奔跑
- InvertedPendulum-v5：倒立摆小车，1 个关节，调试首选（学得快）

和 CartPole 的关键区别：这些环境的状态/动作都是连续的，
状态也不再是 4 个数，而是几十个（位置 + 速度 + 关节角度……）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from core.random_utils import seed_env, set_global_seed


class GaussianActorCriticNetwork(nn.Module):
    """
    高斯策略的 Actor-Critic 网络

    和离散版 ActorCriticNetwork（core/actor_critic.py）对比：
    - 离散版 Actor：状态 → logits（每个离散动作一个分数）
    - 本版 Actor：状态 → 高斯均值 μ（每个连续动作维度一个实数）
      外加一个**与状态无关**的可学习参数 log_std 作为标准差

    用 log_std 而不是直接学 std：保证 std = exp(log_std) 恒为正，
    且 log_std 可以无约束地做梯度下降。

    Actor / Critic 仍是两套独立网络（原因见 actor_critic.py 的注释：
    共享网络时 Critic 的大梯度会压制 Actor 的梯度）。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Actor 网络：状态 → 高斯均值 μ(s)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 各动作维度共享的可学习 log 标准差（初始 std = exp(0) = 1，
        # 训练初期大探索，学会后自动收缩）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic 网络：状态 → 状态价值 V(s)（一个标量）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """返回 (高斯均值 μ, 标准差 σ, 状态价值 V(s))"""
        mean = self.actor(x)
        std = torch.exp(self.log_std)  # 广播到 batch 维度
        value = self.critic(x)
        return mean, std, value


class PPOContinuousAgent:
    """
    连续动作版 PPO 智能体

    和离散版 PPOAgent（core/ppo.py）的两大区别：
    1. 高斯策略：action 是 np 向量，log_prob 对动作维度求和
    2. 按步数攒批：额外记录每步的回合结束标记和结束时的自举价值，
       使一批数据可以跨多个回合（GAE 在回合边界处截断，不向后传播）

    采样 API：
        action = agent.select_action(state)          # 记录 state/action/log_prob/value
        agent.store_reward(reward)                    # 普通步
        agent.store_reward(reward, episode_end=True, end_value=0.0)      # terminated
        agent.store_reward(reward, episode_end=True, end_value=V(final)) # truncated
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_low=None,
        action_high=None,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        update_epochs=10,
        minibatch_size=64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        # 连续动作里"熵"的含义和离散一样（分布越随机熵越大），但高斯熵可正可负，
        # 数值尺度和离散完全不同，调参困难 —— 连续 PPO 通常直接关掉熵奖励（置 0），
        # 靠可学习的 σ 自动维持探索，效果一样好
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.lr0 = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = GaussianActorCriticNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # 动作裁剪范围（不传给环境前的动作做硬裁剪；None 表示不裁剪）
        self.action_low = None if action_low is None else np.asarray(action_low)
        self.action_high = None if action_high is None else np.asarray(action_high)

        # Critic 预热：前 actor_warmup_updates 次更新只练 Critic、冻结 Actor。
        # 用途：任务改了（比如加了指令维度）但策略是预训练来的 —— Critic 需要时间
        # 学会"含指令的状态值多少"，否则它预测不准 → 优势被指令运气主导 →
        # Actor 被乱梯度带崩（指令跟随例子实测需要）
        self.actor_warmup_updates = 0
        self._n_updates_done = 0

        # 一个批次内的记录（采样阶段填充，update 时消费；可跨多个回合）
        self.states = []  # 每步的状态（更新时要重新前向用）
        self.actions = []  # 每步采的动作（np 数组，action_dim 维）
        self.log_probs = []  # 旧策略下该动作的 log 概率（常数，ratio 的分母）
        self.values = []  # 采样时 Critic 估的 V(s)（用来算 GAE）
        self.rewards = []  # 每步奖励
        self.episode_ends = []  # 每步是否是回合最后一步（terminated 或 truncated）
        self.end_values = []  # 回合结束时的自举价值：terminated→0，truncated→V(末状态)

    def select_action(self, state):
        """
        按高斯策略采样一个连续动作向量

        和离散版的区别：
        - 离散版返回 int（动作编号）；本版返回 np 数组（每个关节的力）
        - 高斯各维度独立，log_prob 要对动作维度**求和**得到联合 log 概率
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():  # 采样阶段不需要梯度
            mean, std, value = self.net(state_t)
            dist = Normal(mean, std)
            action = dist.sample()
            # Normal 把每个维度当成独立事件，log_prob 输出 [1, action_dim]，
            # 联合概率 = 各维度概率之积 → log 后变成各维度 log 概率之和
            log_prob = dist.log_prob(action).sum(dim=-1)

        action_np = action.squeeze(0).cpu().numpy()

        self.states.append(np.asarray(state, dtype=np.float32))
        # 关键：记录**未裁剪**的原始采样动作 —— 上面的 log_prob 是在它身上
        # 算的，update 时新旧 log_prob 评估的是同一个动作，ratio 才有意义。
        # 如果存裁剪后的动作，σ 大的时候每步都有大量维度被裁到边界上，
        # old/new log_prob 评估的不是同一个动作，重要性比值被污染
        self.actions.append(action_np.astype(np.float32))
        self.log_probs.append(log_prob.squeeze())  # 旧 log 概率（常数）
        self.values.append(value.squeeze())  # 旧 V(s)（常数）

        # 只有发给环境的动作才裁剪到执行器范围（机器人关节有物理上限；
        # MuJoCo 内部也会按 ctrlrange 截断，这里算双保险）
        if self.action_low is not None:
            return np.clip(action_np, self.action_low, self.action_high)
        return action_np

    def store_reward(self, reward, episode_end=False, end_value=0.0):
        """
        记录一步奖励，以及该步是否结束了回合

        参数:
            reward: 本步奖励
            episode_end: 该步是否是回合最后一步（terminated 或 truncated）
            end_value: 回合结束时的价值自举项 —— terminated 传 0（游戏真结束了，
                       后面没有价值），truncated 传 V(末状态)（只是被截断，
                       后面的价值用 Critic 估计）。仅在 episode_end=True 时生效
        """
        self.rewards.append(reward)
        self.episode_ends.append(bool(episode_end))
        self.end_values.append(float(end_value))

    def anneal_lr(self, frac):
        """按训练进度线性把学习率从 lr0 降到 lr0 的 1/10（和离散版一致）"""
        frac = min(max(frac, 0.0), 1.0)
        lr = self.lr0 * (1.0 - 0.9 * frac)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, last_state=None, bootstrap=False):
        """
        攒满一批后，用裁剪目标把这批数据反复训练 K 个 epoch

        和离散版的差别在 GAE：这批数据可能跨多个回合，所以
        - 回合结束步的 next_value 用 end_value（terminated→0，truncated→V(末状态)）
        - 回合结束步的优势不再向后传播（gae 递推乘 cont=0 截断）
        - 批次末尾若停在回合中途（bootstrap=True），用 V(last_state) 自举

        返回: (total_loss, actor_loss, critic_loss)，是所有 minibatch 更新的平均
        """
        if not self.rewards:
            return 0.0, 0.0, 0.0

        device = self.device
        states = torch.FloatTensor(np.array(self.states)).to(device)
        # 动作是 action_dim 维向量：[T, action_dim]（离散版是 [T] 的 LongTensor）
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.stack(self.log_probs).to(device)  # [T]，常数
        values = torch.stack(self.values).to(device)  # [T]，常数
        rewards_t = torch.FloatTensor(self.rewards).to(device)
        ends = torch.tensor(self.episode_ends, dtype=torch.bool, device=device)
        end_vals = torch.FloatTensor(self.end_values).to(device)

        # ===== 第一步：GAE 算优势 A_t 和回报目标 returns =====
        # 批次末尾的自举价值：停在回合中途 → V(last_state)；否则用不到
        if bootstrap and last_state is not None:
            with torch.no_grad():
                s = torch.FloatTensor(last_state).unsqueeze(0).to(device)
                _, _, last_v = self.net(s)
                last_v = last_v.squeeze().reshape(1)
        else:
            last_v = torch.zeros(1, device=device)
        next_values = torch.cat([values[1:], last_v])
        # 回合结束步的 next_value 用采样时记下的自举价值（跨回合不能取 values[t+1]，那是新回合的起点）
        next_values = torch.where(ends, end_vals, next_values)
        cont = 1.0 - ends.float()  # 回合结束后优势不再向后传播

        deltas = rewards_t + self.gamma * next_values - values
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for i in range(len(deltas) - 1, -1, -1):
            gae = deltas[i] + self.gamma * self.gae_lambda * cont[i] * gae
            advantages[i] = gae
        returns = advantages + values

        # ===== 第二步：标准化优势 =====
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===== 第三步：K 个 epoch × minibatch 复用这批数据 =====
        self._n_updates_done += 1
        train_actor = self._n_updates_done > self.actor_warmup_updates
        n = len(rewards_t)
        indices = np.arange(n)
        sum_total, sum_actor, sum_critic, n_updates = 0.0, 0.0, 0.0, 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb = indices[start : start + self.minibatch_size]
                mb_idx = torch.LongTensor(mb).to(device)

                # 用新策略重新前向（PPO 数据复用的关键）
                mean, std, value_pred = self.net(states[mb_idx])
                dist = Normal(mean, std)
                # 同样要对动作维度求和：[mb, action_dim] → [mb]
                new_log_probs = dist.log_prob(actions[mb_idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                mb_adv = advantages[mb_idx]

                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])

                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(
                    value_pred.squeeze(-1), returns[mb_idx]
                )

                if train_actor:
                    loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy
                    )
                else:
                    # Critic 预热期：只回归价值，Actor（含 log_std）不更新
                    loss = self.value_coef * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                sum_total += float(loss.item())
                sum_actor += float(actor_loss.item())
                sum_critic += float(critic_loss.item())
                n_updates += 1

        # 清空批次记录
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.episode_ends = []
        self.end_values = []

        n_updates = max(n_updates, 1)
        return sum_total / n_updates, sum_actor / n_updates, sum_critic / n_updates


def train_ppo_continuous(
    env,
    agent,
    total_steps=1_000_000,
    batch_size=2048,
    on_update_end=None,
    should_stop=None,
):
    """
    训练连续动作版 PPO 智能体（按步数攒批，工业界做法）

    和离散版 train_ppo 的关键区别：不再"一回合更新一次"，而是
    跨回合连续采样、攒满 batch_size 步更新一次。回合边界信息通过
    store_reward 的 episode_end / end_value 传给 GAE。

    参数:
        env: Gymnasium 环境（MuJoCo 连续控制）
        agent: PPOContinuousAgent 实例
        total_steps: 总采样步数（注意：是步数，不是回合数）
        batch_size: 每次更新攒多少步
        on_update_end: 回调 (update_idx, 批内平均回合奖励, 批内平均回合步数, loss, 已采样步数)

    返回（长度都等于更新次数 ≈ total_steps / batch_size）:
        update_rewards: 每次更新批内完成回合的平均总奖励
        update_steps:   每次更新批内完成回合的平均步数
        total_losses, actor_losses, critic_losses
    """
    update_rewards = []
    update_steps = []
    total_losses = []
    actor_losses = []
    critic_losses = []

    state, _ = env.reset()
    global_step = 0
    update_idx = 0

    while global_step < total_steps:
        if should_stop and should_stop():
            break
        agent.anneal_lr(global_step / total_steps)

        # ===== 收集 batch_size 步（跨回合，回合结束自动重置继续采）=====
        ep_rewards_in_batch = []  # 这批里完整跑完的回合的总奖励
        ep_steps_in_batch = []
        ep_reward, ep_len = 0.0, 0

        for _ in range(batch_size):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated:
                # 真结束（摔倒/失败）：末状态价值按 0 算
                agent.store_reward(reward, episode_end=True, end_value=0.0)
            elif truncated:
                # 只是被步数上限截断：用 V(末状态) 自举，采样时就地算好存下
                with torch.no_grad():
                    s = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                    _, _, v = agent.net(s)
                agent.store_reward(reward, episode_end=True, end_value=float(v.item()))
            else:
                agent.store_reward(reward)

            ep_reward += reward
            ep_len += 1
            global_step += 1
            state = next_state

            if done:
                ep_rewards_in_batch.append(ep_reward)
                ep_steps_in_batch.append(ep_len)
                ep_reward, ep_len = 0.0, 0
                state, _ = env.reset()

        # ===== 更新（批次末尾若停在回合中途，用 V(当前状态) 自举；
        #       若恰好结束，GAE 里的 ends 标记会覆盖，互不影响）=====
        t_loss, a_loss, c_loss = agent.update(last_state=state, bootstrap=True)
        update_idx += 1

        # 批内可能没有完整回合（长存活回合跨批），此时记 0 仅占位
        update_rewards.append(
            float(np.mean(ep_rewards_in_batch)) if ep_rewards_in_batch else 0.0
        )
        update_steps.append(
            float(np.mean(ep_steps_in_batch)) if ep_steps_in_batch else 0.0
        )
        total_losses.append(t_loss)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        if on_update_end:
            on_update_end(
                update_idx, update_rewards[-1], update_steps[-1], t_loss, global_step
            )

    return update_rewards, update_steps, total_losses, actor_losses, critic_losses


def run_ppo_continuous_experiment(
    env_id="Ant-v5",
    total_steps=1_000_000,
    batch_size=2048,
    hidden_dim=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    update_epochs=10,
    minibatch_size=64,
    n_runs=1,
    seed=None,
    env_kwargs=None,
    on_update_end=None,
    should_stop=None,
):
    """
    运行连续动作 PPO 实验，返回 JSON 可序列化的 dict

    供 Web API 调用。avg_rewards / avg_steps 是按"更新批次"记录的序列
    （批内完成回合的平均值），不再是按回合记录。
    """
    import gymnasium as gym

    set_global_seed(seed)
    n_updates_est = max(1, total_steps // batch_size)
    all_rewards = np.zeros(n_updates_est)
    all_steps = np.zeros(n_updates_est)
    all_losses = np.zeros(n_updates_est)
    all_actor_losses = np.zeros(n_updates_est)
    all_critic_losses = np.zeros(n_updates_est)

    for run_idx in range(n_runs):
        if should_stop and should_stop():
            break
        env = gym.make(env_id, **(env_kwargs or {}))
        seed_env(env, None if seed is None else seed + run_idx)

        agent = PPOContinuousAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            value_coef=value_coef,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        u_rewards, u_steps, u_losses, u_actor, u_critic = train_ppo_continuous(
            env,
            agent,
            total_steps=total_steps,
            batch_size=batch_size,
            should_stop=should_stop,
            on_update_end=(
                lambda idx, r, s, loss, step: (
                    on_update_end(run_idx, idx, r, s, loss, step)
                    if on_update_end
                    else None
                )
            ),
        )

        all_rewards[: len(u_rewards)] += np.array(u_rewards)
        all_steps[: len(u_steps)] += np.array(u_steps)
        all_losses[: len(u_losses)] += np.array(u_losses)
        all_actor_losses[: len(u_actor)] += np.array(u_actor)
        all_critic_losses[: len(u_critic)] += np.array(u_critic)

        env.close()

    avg_rewards = (all_rewards / n_runs).tolist()
    avg_steps = (all_steps / n_runs).tolist()
    avg_losses = (all_losses / n_runs).tolist()
    avg_actor_losses = (all_actor_losses / n_runs).tolist()
    avg_critic_losses = (all_critic_losses / n_runs).tolist()

    tail = avg_rewards[-max(1, len(avg_rewards) // 10) :]
    head = avg_rewards[: max(1, len(avg_rewards) // 10)]

    return {
        "env_id": env_id,
        "total_steps": total_steps,
        "avg_rewards": avg_rewards,
        "avg_steps": avg_steps,
        "avg_losses": avg_losses,
        "avg_actor_losses": avg_actor_losses,
        "avg_critic_losses": avg_critic_losses,
        "summary": {
            "final_avg_reward": round(float(np.mean(tail)), 1),
            "initial_avg_reward": round(float(np.mean(head)), 1),
            "max_reward": round(float(np.max(avg_rewards)), 1),
        },
        "seed": seed,
        "params": {
            "env_id": env_id,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "value_coef": value_coef,
            "update_epochs": update_epochs,
            "minibatch_size": minibatch_size,
            "n_runs": n_runs,
        },
    }
