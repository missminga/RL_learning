"""
Policy Gradient (REINFORCE) + CartPole —— 直接学策略

=== 回顾：DQN 的思路 ===
上一个例子中，DQN 学的是 Q(s,a)——"在状态 s 做动作 a 有多好"。
然后选动作时，挑 Q 值最大的。这叫"基于价值"的方法 (Value-Based)。

=== 新思路：直接学策略 ===
能不能跳过 Q 值，直接学"在状态 s 下应该选哪个动作"？
这就是"基于策略"的方法 (Policy-Based)。

策略网络 π(a|s) 输出的不再是 Q 值，而是一个概率分布：
  π(左推|s) = 0.7, π(右推|s) = 0.3
然后按这个概率随机选一个动作。

=== REINFORCE 算法 ===
最经典的 Policy Gradient 算法，核心思想极其简单：
  如果一个动作最终带来了高回报 → 增大这个动作的概率
  如果一个动作最终带来了低回报 → 减小这个动作的概率

具体来说：
  1. 跑完一整个回合，得到轨迹 (s₀,a₀,r₀), (s₁,a₁,r₁), ...
  2. 计算每步的折扣回报 Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...
  3. 更新：θ ← θ + α * Gₜ * ∇θ log π(aₜ|sₜ)

=== 对比 DQN vs REINFORCE ===
┌─────────────┬───────────────────┬───────────────────┐
│             │ DQN               │ REINFORCE         │
├─────────────┼───────────────────┼───────────────────┤
│ 学什么       │ Q 值（价值函数）   │ 策略（概率分布）   │
│ 选动作       │ ε-贪心 + argmax Q │ 按概率采样         │
│ 更新时机     │ 每一步（在线）     │ 跑完一整个回合     │
│ 经验回放     │ 需要              │ 不需要             │
│ 目标网络     │ 需要              │ 不需要             │
│ 连续动作空间 │ 不行（只能离散）   │ 可以！             │
└─────────────┴───────────────────┴───────────────────┘

REINFORCE 最大的优势：能处理连续动作空间（比如控制机器人关节角度）。
DQN 只能输出有限个 Q 值，没法处理连续动作。

=== CartPole 环境 ===
和上一个例子相同：
- 状态：[小车位置, 小车速度, 杆角度, 杆角速度]（4 维连续）
- 动作：左推(0) / 右推(1)
- 奖励：每活一步 +1
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.policy_gradient import REINFORCEAgent, train_reinforce


def plot_results(episode_rewards, episode_steps, losses):
    """画三张图：奖励曲线、步数曲线、损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    window = 20

    # --- 图1: 每回合总奖励 ---
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, color="#5470c6", label="每回合奖励")
    if len(episode_rewards) >= window:
        smoothed = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(episode_rewards)), smoothed,
            color="#ee6666", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax1.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准 (475)")
    ax1.set_xlabel("回合 (Episode)")
    ax1.set_ylabel("总奖励")
    ax1.set_title("REINFORCE 学习曲线")
    ax1.legend()

    # --- 图2: 每回合步数 ---
    ax2 = axes[1]
    ax2.plot(episode_steps, alpha=0.3, color="#91cc75", label="每回合步数")
    if len(episode_steps) >= window:
        smoothed_steps = np.convolve(
            episode_steps, np.ones(window) / window, mode="valid"
        )
        ax2.plot(
            range(window - 1, len(episode_steps)), smoothed_steps,
            color="#fac858", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax2.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准")
    ax2.set_xlabel("回合 (Episode)")
    ax2.set_ylabel("步数")
    ax2.set_title("每回合存活步数")
    ax2.legend()

    # --- 图3: 训练损失 ---
    ax3 = axes[2]
    ax3.plot(losses, alpha=0.3, color="#fc8452", label="每回合损失")
    if len(losses) >= window:
        smoothed_loss = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        ax3.plot(
            range(window - 1, len(losses)), smoothed_loss,
            color="#9a60b4", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax3.set_xlabel("回合 (Episode)")
    ax3.set_ylabel("策略梯度损失")
    ax3.set_title("训练损失（注意：损失方向和 DQN 不同）")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("examples/04_policy_gradient_result.png", dpi=100)
    print("\n图表已保存到 examples/04_policy_gradient_result.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 55)
    print("Policy Gradient (REINFORCE) + CartPole 实验")
    print("=" * 55)

    # ===== 创建环境 =====
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    print(f"\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}")
    print(f"  动作数量: {action_dim}")

    # ===== 训练参数 =====
    # 注意：REINFORCE 的参数比 DQN 简单很多！
    # 不需要：经验回放容量、批大小、目标网络更新频率、ε 相关参数
    episodes = 500
    hidden_dim = 128
    lr = 1e-3
    gamma = 0.99

    print(f"\n训练参数：")
    print(f"  回合数: {episodes}  （比 DQN 多一些，因为 REINFORCE 方差较大）")
    print(f"  网络隐藏层: {hidden_dim} 个神经元")
    print(f"  学习率: {lr}")
    print(f"  折扣因子 γ: {gamma}")
    print(f"\n注意：没有 ε、没有经验回放、没有目标网络——比 DQN 简洁多了！")

    # ===== 创建智能体 =====
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
    )

    # ===== 开始训练 =====
    print(f"\n开始训练（设备: {agent.device}）...")

    def on_episode_end(ep, reward, steps, loss):
        if (ep + 1) % 50 == 0:
            print(f"  回合 {ep+1:3d}/{episodes}  "
                  f"奖励: {reward:6.1f}  步数: {steps:3d}  "
                  f"损失: {loss:8.2f}")

    episode_rewards, episode_steps, losses = train_reinforce(
        env, agent, episodes=episodes, max_steps=500,
        on_episode_end=on_episode_end,
    )
    print("训练完成！")

    # ===== 结果分析 =====
    last_50_reward = np.mean(episode_rewards[-50:])
    last_50_steps = np.mean(episode_steps[-50:])
    max_reward = np.max(episode_rewards)

    print(f"\n最后 50 回合平均奖励: {last_50_reward:.1f}")
    print(f"最后 50 回合平均步数: {last_50_steps:.1f}")
    print(f"单回合最高奖励: {max_reward:.1f}")

    # 检查是否解决
    solved = False
    if len(episode_rewards) >= 100:
        for i in range(99, len(episode_rewards)):
            window_avg = np.mean(episode_rewards[max(0, i - 99):i + 1])
            if window_avg >= 475:
                print(f"\nCartPole 已解决！在第 {i - 99} 回合达到标准")
                solved = True
                break
    if not solved:
        print(f"\n未完全解决 CartPole（REINFORCE 方差较大，可以多跑几次试试）")

    # ===== 演示一个回合 =====
    print("\n--- 用训练好的智能体演示一回合 ---")
    state, _ = env.reset()
    total_reward = 0
    for step in range(500):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    # 演示结束后清空 agent 的记录（不需要更新）
    agent.log_probs = []
    agent.rewards = []
    print(f"  演示回合: 存活 {step + 1} 步, 总奖励 {total_reward:.0f}")

    env.close()

    # ===== 画图 =====
    plot_results(episode_rewards, episode_steps, losses)

    # ===== 教学总结 =====
    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. 策略梯度: 直接学"选哪个动作的概率"，不用先学 Q 值
2. REINFORCE: 跑完整个回合 → 算回报 → 强化好动作、抑制差动作
3. 更简洁: 不需要经验回放、目标网络、ε-贪心
4. 天然探索: 按概率选动作，自带随机性，不需要 ε
5. 回报标准化: 减均值除标准差，减少梯度方差

对比 DQN vs REINFORCE：
  DQN:       学 Q 值 → 选最大的 → 间接得到策略
  REINFORCE: 直接学策略 → 按概率选动作

REINFORCE 的缺点：
  - 方差大（每次跑的回合不同，回报波动很大）
  - 必须跑完整个回合才能更新（在线学习不行）
  - 样本效率低（每个经验只用一次）

下一步：结合 Value-Based 和 Policy-Based 的优点 → Actor-Critic 方法：
  - Actor（演员）：策略网络，决定选什么动作
  - Critic（评委）：价值网络，评估动作好不好
  这就是 A2C、PPO 等现代算法的基础！
""")


if __name__ == "__main__":
    main()
