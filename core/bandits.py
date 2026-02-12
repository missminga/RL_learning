"""
多臂老虎机核心算法模块

提供 MultiArmedBandit 环境、ε-贪心策略、实验运行函数。
CLI 脚本和 Web API 共用此模块。
"""

import numpy as np


class MultiArmedBandit:
    """多臂老虎机环境"""

    def __init__(self, k=10):
        """
        初始化 k 台老虎机
        每台机器的真实期望奖励从标准正态分布中随机生成
        """
        self.k = k
        # 每台老虎机的真实期望奖励（这个值智能体是不知道的）
        self.q_true = np.random.randn(k)

    def pull(self, action):
        """
        拉第 action 台老虎机，返回奖励
        奖励 = 该机器的真实期望 + 随机噪声
        """
        return self.q_true[action] + np.random.randn()


def epsilon_greedy(q_estimates, epsilon):
    """
    ε-贪心策略：选择一个动作

    参数:
        q_estimates: 当前对每台机器的奖励估计值
        epsilon: 探索的概率（0~1之间）

    返回:
        选择的动作（哪台机器）
    """
    if np.random.random() < epsilon:
        # 探索：随机选一台
        return np.random.randint(len(q_estimates))
    else:
        # 利用：选估计值最大的那台
        return np.argmax(q_estimates)


def run_experiment(k=10, steps=1000, epsilon=0.1):
    """
    运行一次实验

    参数:
        k: 老虎机的数量
        steps: 总共拉多少次
        epsilon: 探索概率

    返回:
        rewards: 每一步获得的奖励
        optimal_actions: 每一步是否选了最优的机器
    """
    bandit = MultiArmedBandit(k)

    # 对每台机器的奖励估计值，初始都为 0
    q_estimates = np.zeros(k)
    # 每台机器被拉的次数
    action_counts = np.zeros(k)

    # 记录每一步的结果
    rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    # 真正最好的那台机器
    best_action = np.argmax(bandit.q_true)

    for step in range(steps):
        # 1. 用ε-贪心策略选择动作
        action = epsilon_greedy(q_estimates, epsilon)

        # 2. 拉老虎机，获得奖励
        reward = bandit.pull(action)

        # 3. 更新该机器的估计值（增量式平均）
        action_counts[action] += 1
        # 新估计 = 旧估计 + (1/次数) * (新奖励 - 旧估计)
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

        # 记录
        rewards[step] = reward
        optimal_actions[step] = (action == best_action)

    return rewards, optimal_actions


def run_comparison(epsilons, k=10, steps=1000, n_runs=200):
    """
    对比多种 ε 值的实验结果，返回 JSON 可序列化的 dict

    参数:
        epsilons: ε 值列表，如 [0, 0.01, 0.1]
        k: 老虎机数量
        steps: 每次实验步数
        n_runs: 重复实验次数（取平均）

    返回:
        dict: {
            "epsilons": [...],
            "rewards": {eps: [平均奖励列表]},
            "optimal_pct": {eps: [最优动作比例列表, 0~100]},
            "summary": [{eps, avg_reward, optimal_pct}, ...]
        }
    """
    all_rewards = {}
    all_optimal = {}

    for eps in epsilons:
        rewards_sum = np.zeros(steps)
        optimal_sum = np.zeros(steps)

        for _ in range(n_runs):
            rewards, optimal = run_experiment(k, steps, eps)
            rewards_sum += rewards
            optimal_sum += optimal

        all_rewards[eps] = rewards_sum / n_runs
        all_optimal[eps] = (optimal_sum / n_runs) * 100  # 转为百分比

    # 构建摘要：最后 100 步的平均值
    summary = []
    for eps in epsilons:
        tail = min(100, steps)
        summary.append({
            "epsilon": eps,
            "avg_reward": round(float(np.mean(all_rewards[eps][-tail:])), 3),
            "optimal_pct": round(float(np.mean(all_optimal[eps][-tail:])), 1),
        })

    # 转为 JSON 可序列化的格式（numpy → list）
    # 用 _eps_key 确保 0.0 → "0"、0.10 → "0.1" 等，与前端 String(eps) 对齐
    def _eps_key(eps):
        return str(int(eps)) if eps == int(eps) else str(eps)

    rewards_dict = {_eps_key(eps): all_rewards[eps].tolist() for eps in epsilons}
    optimal_dict = {_eps_key(eps): all_optimal[eps].tolist() for eps in epsilons}

    return {
        "epsilons": epsilons,
        "rewards": rewards_dict,
        "optimal_pct": optimal_dict,
        "summary": summary,
    }
