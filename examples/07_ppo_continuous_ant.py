"""
连续动作 PPO + MuJoCo 四足机器人 —— 从"按按钮"到"拧关节"

=== 回顾：CartPole 离真机器人还差什么 ===

06 号例子里，PPO 玩 CartPole 的动作是"左推 / 右推"二选一（离散动作）。
但真实机器人没有按钮可按：四足机器人有 8~12 个关节，每个关节要输出
一个**连续数值**的扭矩（比如 0.35 N·m）。离散分布描述不了这种动作。

=== 本例的关键一跃：高斯策略 ===

把策略分布从 Categorical 换成 Normal（高斯），一行公式看懂：

    离散：a ~ Categorical(softmax(logits))     ← 从有限个动作里挑一个
    连续：a ~ Normal(μ(s), σ)                  ← 每个关节维度采一个实数

Actor 网络输出 μ(s)（最想出的力），标准差 σ 是可学习参数（探索的胆子）。
**PPO 算法本身一行不改**——GAE、clip、minibatch 复用全部原样保留，
这就是"策略分布"和"优化算法"解耦的威力。

=== 按步数攒批：工业界 PPO 的标配 ===

离散版是"一回合更新一次"。但 Ant 训练初期每回合只活 20~30 步就摔倒，
回合制更新 = 每次只拿几十条数据就学一次，batch 太小根本学不动。
工业界（CleanRL、Isaac Lab 的 rsl_rl）的做法：**跨回合边界连续采样，
攒满 batch_size 步（如 2048）才更新一次**，数据量恒定、梯度稳定。

代价是 GAE 要处理"一批数据跨多个回合"：每步记录回合是否结束，
terminated → 末状态价值算 0，truncated → 用 V(末状态) 自举，
批次停在回合中途 → 用 V(当前状态) 自举。

=== MuJoCo：真正的物理仿真器 ===

本例环境是 Ant-v5（四足蚂蚁机器人），由 MuJoCo 物理引擎驱动：
- 状态：27 维连续向量（躯干姿态/速度 + 关节角度/角速度；本例关掉了默认开启的
  78 维接触力观测 —— 那些数值大、噪声尖刺多，不归一化很难学）
- 动作：8 维连续向量（8 个关节的扭矩，范围 [-1, 1]）
- 奖励：前进得越快奖励越高，活着有存活奖励，动作太猛有惩罚

Ant 的每一步都是 MuJoCo 在解完整的多刚体接触动力学 —— 脚踩地的
摩擦力、关节限位、惯性耦合，全都真实模拟。这就是机器人 locomotion
训练的"幼儿园版真机"。

=== 训练量提示 ===

Ant 需要约 100 万步才能明显走起来（本例默认 1M 步，CPU 上约 10~20 分钟）。
想快速验证流程可以把 ENV_ID 改成 "InvertedPendulum-v5"、total_steps 改小；
想效果更好可以加大到 2M~3M，或放到有 NVIDIA 显卡的机器上跑。
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.ppo_continuous import PPOContinuousAgent, train_ppo_continuous

# matplotlib 默认字体不含中文，指定系统中文字体避免图例显示为方框
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class UprightPostureWrapper(gym.Wrapper):
    """
    姿势约束 wrapper：专治 HalfCheetah "肚皮朝天倒着跑"的 reward hacking

    HalfCheetah 默认奖励只管前进速度、不管姿势，策略会翻出肚皮用背跑
    （这是 OpenAI "Faulty Reward Functions" 博客的经典案例）。
    这里加两条规矩：
    1. 姿势惩罚：奖励减去 penalty·|躯干俯仰角| —— 前倾后倾都扣分
    2. 翻转即终止：|俯仰角| > pitch_limit 判为"摔倒"，回合直接结束
       —— 学 Ant 的 healthy 机制：翻倒 = 死亡，策略自然躲开翻转区

    这就是 reward shaping：想要什么仪态，就把它写进奖励。
    """

    def __init__(self, env, pitch_limit=1.5, penalty=0.1):
        super().__init__(env)
        self.pitch_limit = pitch_limit
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # qpos[2] 是躯干俯仰角：直立≈0，肚皮朝天≈±π（3.14）
        pitch = float(self.env.unwrapped.data.qpos[2])
        reward = reward - self.penalty * abs(pitch)
        if abs(pitch) > self.pitch_limit:
            terminated = True
        return obs, reward, terminated, truncated, info


def plot_results(
    update_rewards, update_steps, actor_losses, critic_losses, out_path, env_label
):
    """画四张图：批内平均回合奖励、平均步数、Actor 损失、Critic 损失"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    window = 20

    # --- 图1: 批内平均回合奖励 ---
    ax1 = axes[0]
    ax1.plot(update_rewards, alpha=0.3, color="#5470c6", label="批内平均回合奖励")
    if len(update_rewards) >= window:
        smoothed = np.convolve(update_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(update_rewards)),
            smoothed,
            color="#ee6666",
            linewidth=2,
            label=f"滑动平均 (窗口={window})",
        )
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("更新次数 (Update)")
    ax1.set_ylabel("平均回合奖励")
    ax1.set_title(f"连续 PPO 学习曲线（{env_label}）")
    ax1.legend()

    # --- 图2: 批内平均回合步数 ---
    ax2 = axes[1]
    ax2.plot(update_steps, alpha=0.3, color="#91cc75", label="批内平均回合步数")
    if len(update_steps) >= window:
        smoothed_steps = np.convolve(
            update_steps, np.ones(window) / window, mode="valid"
        )
        ax2.plot(
            range(window - 1, len(update_steps)),
            smoothed_steps,
            color="#fac858",
            linewidth=2,
            label=f"滑动平均 (窗口={window})",
        )
    ax2.set_xlabel("更新次数 (Update)")
    ax2.set_ylabel("平均步数")
    ax2.set_title("每回合存活步数")
    ax2.legend()

    # --- 图3: Actor 损失 ---
    ax3 = axes[2]
    ax3.plot(actor_losses, alpha=0.3, color="#fc8452", label="Actor 损失")
    if len(actor_losses) >= window:
        smoothed_a = np.convolve(actor_losses, np.ones(window) / window, mode="valid")
        ax3.plot(
            range(window - 1, len(actor_losses)),
            smoothed_a,
            color="#9a60b4",
            linewidth=2,
            label=f"滑动平均 (窗口={window})",
        )
    ax3.set_xlabel("更新次数 (Update)")
    ax3.set_ylabel("Actor 损失")
    ax3.set_title("演员损失（裁剪策略部分）")
    ax3.legend()

    # --- 图4: Critic 损失 ---
    ax4 = axes[3]
    ax4.plot(critic_losses, alpha=0.3, color="#73c0de", label="Critic 损失")
    if len(critic_losses) >= window:
        smoothed_c = np.convolve(critic_losses, np.ones(window) / window, mode="valid")
        ax4.plot(
            range(window - 1, len(critic_losses)),
            smoothed_c,
            color="#3ba272",
            linewidth=2,
            label=f"滑动平均 (窗口={window})",
        )
    ax4.set_xlabel("更新次数 (Update)")
    ax4.set_ylabel("Critic 损失")
    ax4.set_title("评委损失（价值估计误差）")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"\n图表已保存到 {out_path}")


def record_demo_video(agent, env_id, env_kwargs, out_path, episodes=2, upright=False):
    """
    用训练好的策略跑几个回合，录制成 mp4 视频

    注意部署态和训练态的区别：训练时要从 Normal(μ, σ) 里**采样**（保持探索），
    部署时直接取均值 μ 作为动作（不再探索，表现最稳）。
    """
    # 优先用 track 相机（镜头跟着机器人走），环境没有就退回默认相机
    try:
        env = gym.make(
            env_id, render_mode="rgb_array", camera_name="track", **env_kwargs
        )
    except Exception:
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    if upright:
        env = UprightPostureWrapper(env)
    frames = []
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(1000):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                mean, _, _ = agent.net(state_t)  # 部署：取高斯均值，不采样
            action = mean.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
    env.close()
    # MuJoCo locomotion 物理步频 20Hz（0.01s × frame_skip 5），20fps 才是真实速度
    imageio.mimsave(out_path, frames, fps=20)
    print(f"演示视频已保存到 {out_path}（{len(frames)} 帧，{episodes} 个回合）")


def main():
    """主函数"""
    # ===== 命令行参数 =====
    # 环境可选（观测/动作维度自动适配，core/ppo_continuous.py 无需改动）：
    #   Ant-v5:              四足机器人（默认）
    #   HalfCheetah-v5:      二维奔跑猎豹，更容易看到奖励爬升
    #   InvertedPendulum-v5: 倒立摆，调试用（学得最快，10 万步内见效）
    # 用法: uv run python examples/07_ppo_continuous_ant.py --env HalfCheetah-v5
    import argparse

    parser = argparse.ArgumentParser(description="连续动作 PPO + MuJoCo 机器人")
    parser.add_argument("--env", default="Ant-v5", help="Gymnasium 环境 ID")
    parser.add_argument("--steps", type=int, default=1_000_000, help="总采样步数")
    parser.add_argument(
        "--upright",
        action="store_true",
        help="加姿势约束（HalfCheetah 专用：翻转即终止 + 姿势惩罚，治倒着跑）",
    )
    args = parser.parse_args()
    ENV_ID = args.env
    # 产物文件名：07_<env>[_upright]_policy.pt 等，不同配置互不覆盖
    slug = ENV_ID.split("-")[0].lower() + ("_upright" if args.upright else "")

    print("=" * 55)
    print("连续动作 PPO（高斯策略）+ MuJoCo 机器人实验")
    print("=" * 55)

    # ===== 创建环境 =====
    # Ant-v5 默认在观测里加入 78 维脚部接触力（数值大、噪声尖刺多），
    # 没有观测归一化时很难学。关掉它回到经典的 27 维观测（角度+速度）
    env_kwargs = (
        {"include_cfrc_ext_in_observation": False} if ENV_ID.startswith("Ant") else {}
    )
    env = gym.make(ENV_ID, **env_kwargs)
    if args.upright:
        env = UprightPostureWrapper(env)
        print(
            "已启用姿势约束：|躯干俯仰角| > 1.5 rad 判为摔倒（terminated），"
            "并按 0.1×|俯仰角| 扣分"
        )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"\n环境: {ENV_ID}（MuJoCo 物理引擎驱动）")
    print(f"  状态维度: {state_dim}（连续）")
    print(
        f"  动作维度: {action_dim}（连续，范围 "
        f"[{env.action_space.low[0]:.0f}, {env.action_space.high[0]:.0f}]）"
    )

    # ===== 训练参数 =====
    total_steps = args.steps  # 总采样步数（工业界按步数算，不按回合数）
    batch_size = 2048  # 每次更新攒多少步（跨回合）
    hidden_dim = 128
    lr = 3e-4  # 连续控制比离散敏感，学习率小一个量级
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    update_epochs = 10
    minibatch_size = 64

    n_updates = total_steps // batch_size

    print("\n训练参数：")
    print(
        f"  总步数: {total_steps:,}，每批: {batch_size} 步（→ 约 {n_updates} 次更新）"
    )
    print(f"  学习率: {lr}（连续控制用小学习率）")
    print(f"  折扣因子 γ: {gamma}，GAE λ: {gae_lambda}，裁剪范围 ε: {clip_eps}")
    print(f"  数据复用 epoch 数: {update_epochs}，minibatch 大小: {minibatch_size}")
    print("\n和离散版的差异：Categorical → Normal + 按步数攒批，算法核心不变。")

    # ===== 创建智能体 =====
    agent = PPOContinuousAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,  # 关节扭矩下限
        action_high=env.action_space.high,  # 关节扭矩上限
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
    )

    # ===== 开始训练 =====
    print(f"\n开始训练（设备: {agent.device}，物理仿真在 CPU 上）...")
    print("提示：Ant 大约从第 100 次更新（20 万步）开始奖励明显爬升\n")

    def on_update_end(idx, mean_reward, mean_steps, loss, global_step):
        if idx % 10 == 0:
            print(
                f"  更新 {idx:4d}/{n_updates}  步数 {global_step:>9,}  "
                f"批内平均奖励: {mean_reward:8.1f}  "
                f"批内平均存活: {mean_steps:5.1f} 步"
            )

    update_rewards, update_steps, total_losses, actor_losses, critic_losses = (
        train_ppo_continuous(
            env,
            agent,
            total_steps=total_steps,
            batch_size=batch_size,
            on_update_end=on_update_end,
        )
    )
    print("训练完成！")

    # ===== 结果分析 =====
    k = max(1, len(update_rewards) // 10)
    head_reward = np.mean(update_rewards[:k])
    tail_reward = np.mean(update_rewards[-k:])
    max_reward = np.max(update_rewards)

    print(f"\n最初 10% 更新的平均奖励: {head_reward:.1f}")
    print(f"最后 10% 更新的平均奖励: {tail_reward:.1f}")
    print(f"单批最高平均奖励: {max_reward:.1f}")
    if tail_reward > max(head_reward * 1.5, 100):
        print("\n奖励明显爬升，蚂蚁学起来了！")
    else:
        print("\n奖励爬升不明显 —— 可以加到 2M~3M 步，或换 HalfCheetah-v5 试试")

    env.close()

    # ===== 保存策略权重（下次可视化/部署不用重训）=====
    policy_path = f"examples/07_{slug}_policy.pt"
    torch.save(agent.net.state_dict(), policy_path)
    print(f"\n策略权重已保存到 {policy_path}")

    # ===== 录一段演示视频 =====
    record_demo_video(
        agent,
        ENV_ID,
        env_kwargs,
        f"examples/07_{slug}_demo.mp4",
        upright=args.upright,
    )

    # ===== 画图 =====
    plot_results(
        update_rewards,
        update_steps,
        actor_losses,
        critic_losses,
        out_path=f"examples/07_{slug}_result.png",
        env_label=ENV_ID,
    )

    # ===== 教学总结 =====
    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. 高斯策略：连续动作用 Normal(μ(s), σ) 建模，μ 由网络输出，σ 可学习
   —— σ 自动从大（探索）收缩到小（收敛），所以连续 PPO 一般不加熵奖励
2. 换分布不改算法：GAE、clip、minibatch 复用和离散版逐行对应，
   只有 log_prob 要对动作维度求和（各维度独立 → 联合概率是乘积）
3. 按步数攒批：跨回合攒满 batch_size 步再更新，batch 恒定、梯度稳定
   —— 回合边界处 GAE 截断：terminated 价值算 0，truncated 用 V(末状态) 自举
4. MuJoCo 环境：Ant 的每一步都是真实多刚体接触动力学，
   这就是机器人 locomotion 训练的微缩版
5. 动作裁剪：发给环境的动作裁剪到扭矩范围 [-1, 1]（执行器物理上限），
   但训练记录的是未裁剪的原始采样 —— 否则新旧 log_prob 评估的不是同一个
   动作，重要性比值被污染，PPO 学不动（这是本例修过的真实 bug）
6. 训练 vs 部署：训练时从 Normal(μ,σ) 采样（要探索）；部署时直接取 μ
   （不探索，最稳）——演示视频里就是部署态
""")
    print(f"""
本次产物（按环境名区分，互不覆盖）：
  - 学习曲线图: examples/07_{slug}_result.png
  - 策略权重:   examples/07_{slug}_policy.pt（加载即可部署，无需重训）
  - 演示视频:   examples/07_{slug}_demo.mp4（看看你的机器人学会怎么走了！）

对比 CartPole vs Ant：
  CartPole: 状态 4 维、动作 2 选 1、25 万步解决
  Ant:      状态 27 维、动作 8 维连续、100 万步起步 —— 但算法是同一个 PPO

承上启下：
  - 你刚完成了"自定义机器人"的第一块基石：连续控制策略
  - 下一步：学会读/写 MJCF（MuJoCo 的机器人模型 XML），
    把自己 3D 打印的机器人建出来，用同样的代码训练它走路
  - 再往后：Isaac Lab 大规模并行训练、域随机化、sim-to-real
""")


if __name__ == "__main__":
    main()
