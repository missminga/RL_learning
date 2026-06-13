"""
Actor-Critic (A2C) + CartPole —— 演员与评委的协作

=== 回顾：REINFORCE 的痛点 ===
上一个例子 REINFORCE 直接用回报 G_t 来加权更新策略：
  θ ← θ + α * G_t * ∇θ log π(a_t|s_t)
问题是 G_t 把一整条轨迹的随机奖励都加起来，**方差极大**，
表现就是：训练曲线剧烈抖动、时好时坏、收敛慢。

=== 新思路：给策略配一个"评委" ===
A2C（Advantage Actor-Critic）同时训练两个角色：
  - Actor（演员）：策略网络 π(a|s)，负责选动作（和 REINFORCE 一样）
  - Critic（评委）：价值网络 V(s)，负责评估"这个状态大概值多少分"

关键是用"优势 (Advantage)"代替原始回报：
  A(s,a) = G_t - V(s)
意思是："这个动作实际拿到的回报，比这个状态本来平均能拿的好多少？"
  - 比平均好 (A>0) → 增大这个动作的概率
  - 比平均差 (A<0) → 减小这个动作的概率

减去 V(s) 这个 baseline 不改变梯度期望（无偏），却能大幅降低方差——
这就是 A2C 比 REINFORCE 稳的根本原因。

=== 两个损失一起优化 ===
  actor_loss  = -Σ log π(a_t|s_t) * A_t      （演员：朝优势大的方向调策略）
  critic_loss =  Σ (V(s_t) - G_t)²           （评委：让估值逼近真实回报）
  entropy     =  鼓励策略保持随机性，防止过早收敛
  total = actor_loss + 0.5*critic_loss - 0.01*entropy

=== 三种方法一览 ===
┌─────────────┬──────────────┬──────────────┬────────────────────┐
│             │ DQN          │ REINFORCE    │ A2C                │
├─────────────┼──────────────┼──────────────┼────────────────────┤
│ 学什么       │ Q(s,a)       │ π(a|s)       │ π(a|s) + V(s)      │
│ 评估信号     │ TD 目标       │ 回报 G_t     │ 优势 G_t - V(s)    │
│ 方差         │ 低（有偏）    │ 高           │ 中（兼顾偏差方差）  │
│ 网络         │ 策略+目标网络 │ 1 个策略网络  │ 1 网络两个头        │
└─────────────┴──────────────┴──────────────┴────────────────────┘

A2C 是 PPO、A3C、SAC 等现代算法的基础。

=== CartPole 环境 ===
和前两个例子相同：
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
from core.actor_critic import A2CAgent, train_a2c


def plot_results(episode_rewards, episode_steps, actor_losses, critic_losses):
    """画四张图：奖励、步数、Actor 损失、Critic 损失"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    window = 20

    # --- 图1: 每回合总奖励 ---
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, color="#5470c6", label="每回合奖励")
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(episode_rewards)), smoothed,
            color="#ee6666", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax1.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准 (475)")
    ax1.set_xlabel("回合 (Episode)")
    ax1.set_ylabel("总奖励")
    ax1.set_title("A2C 学习曲线")
    ax1.legend()

    # --- 图2: 每回合步数 ---
    ax2 = axes[1]
    ax2.plot(episode_steps, alpha=0.3, color="#91cc75", label="每回合步数")
    if len(episode_steps) >= window:
        smoothed_steps = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        ax2.plot(
            range(window - 1, len(episode_steps)), smoothed_steps,
            color="#fac858", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax2.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准")
    ax2.set_xlabel("回合 (Episode)")
    ax2.set_ylabel("步数")
    ax2.set_title("每回合存活步数")
    ax2.legend()

    # --- 图3: Actor 损失 ---
    ax3 = axes[2]
    ax3.plot(actor_losses, alpha=0.3, color="#fc8452", label="Actor 损失")
    if len(actor_losses) >= window:
        smoothed_a = np.convolve(actor_losses, np.ones(window) / window, mode="valid")
        ax3.plot(
            range(window - 1, len(actor_losses)), smoothed_a,
            color="#9a60b4", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax3.set_xlabel("回合 (Episode)")
    ax3.set_ylabel("Actor 损失")
    ax3.set_title("演员损失（策略部分）")
    ax3.legend()

    # --- 图4: Critic 损失 ---
    ax4 = axes[3]
    ax4.plot(critic_losses, alpha=0.3, color="#73c0de", label="Critic 损失")
    if len(critic_losses) >= window:
        smoothed_c = np.convolve(critic_losses, np.ones(window) / window, mode="valid")
        ax4.plot(
            range(window - 1, len(critic_losses)), smoothed_c,
            color="#3ba272", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax4.set_xlabel("回合 (Episode)")
    ax4.set_ylabel("Critic 损失")
    ax4.set_title("评委损失（价值估计误差）")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("examples/05_actor_critic_result.png", dpi=100)
    print("\n图表已保存到 examples/05_actor_critic_result.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 55)
    print("Actor-Critic (A2C) + CartPole 实验")
    print("=" * 55)

    # ===== 创建环境 =====
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    print("\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}")
    print(f"  动作数量: {action_dim}")

    # ===== 训练参数 =====
    episodes = 500
    hidden_dim = 128
    lr = 1e-3
    gamma = 0.99
    value_coef = 0.5      # Critic 损失权重
    entropy_coef = 0.01   # 熵奖励权重

    print("\n训练参数：")
    print(f"  回合数: {episodes}")
    print(f"  网络隐藏层: {hidden_dim} 个神经元（Actor 与 Critic 共享躯干）")
    print(f"  学习率: {lr}")
    print(f"  折扣因子 γ: {gamma}")
    print(f"  Critic 损失权重: {value_coef}，熵奖励权重: {entropy_coef}")
    print("\n相比 REINFORCE：多了一个 Critic，用'优势'代替'回报'，方差更小、更稳定。")

    # ===== 创建智能体 =====
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )

    # ===== 开始训练 =====
    print(f"\n开始训练（设备: {agent.device}）...")

    def on_episode_end(ep, reward, steps, loss):
        if (ep + 1) % 50 == 0:
            print(f"  回合 {ep+1:3d}/{episodes}  "
                  f"奖励: {reward:6.1f}  步数: {steps:3d}  "
                  f"总损失: {loss:8.2f}")

    episode_rewards, episode_steps, total_losses, actor_losses, critic_losses = train_a2c(
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
        print("\n未完全解决 CartPole（可以多跑几次或增加回合数试试）")

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
    agent.values = []
    agent.rewards = []
    agent.entropies = []
    print(f"  演示回合: 存活 {step + 1} 步, 总奖励 {total_reward:.0f}")

    env.close()

    # ===== 画图 =====
    plot_results(episode_rewards, episode_steps, actor_losses, critic_losses)

    # ===== 教学总结 =====
    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. Actor-Critic: 演员(策略)负责动作，评委(价值)负责评估，两者一起训练
2. 优势函数 A = G_t - V(s): 用"相对回报"代替"绝对回报"，大幅降低方差
3. baseline 免费午餐: 减去 V(s) 不改变梯度期望(无偏)，只降方差
4. 两个损失: actor_loss(策略) + critic_loss(回归) + 熵奖励(探索)
5. 共享躯干: Actor 和 Critic 共用特征提取层，省参数、互相促进

对比 REINFORCE vs A2C：
  REINFORCE: 用整条轨迹的回报 G_t 加权 → 方差大、抖动剧烈
  A2C:       用优势 G_t - V(s) 加权    → 方差小、训练更稳

承上启下：
  - 往前看：A2C = REINFORCE(基于策略) + 价值估计(借鉴 DQN 的思想)
  - 往后看：把 A2C 的更新加上"裁剪"限制每次步子别太大 → 就是 PPO
    PPO 是目前最主流的策略优化算法（ChatGPT 的 RLHF 也用它）
""")


if __name__ == "__main__":
    main()
