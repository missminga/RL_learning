"""PPO (近端策略优化) 模块单元测试"""

import numpy as np
import pytest
import torch
import gymnasium as gym

from core.ppo import (
    PPOAgent,
    train_ppo,
    run_ppo_experiment,
)


# ===== PPOAgent 测试 =====


class TestPPOAgent:
    """PPO 智能体测试"""

    @pytest.fixture
    def agent(self):
        return PPOAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=32,
            lr=1e-3,
            gamma=0.99,
            update_epochs=3,
            minibatch_size=16,
        )

    def test_select_action_range(self, agent):
        """选出的动作应在有效范围内"""
        state = np.array([0.0, 0.1, -0.05, 0.2], dtype=np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert action in [0, 1]

    def test_select_action_records_history(self, agent):
        """select_action 应同时记录 state、action、log_prob、value"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert len(agent.states) == 0
        agent.select_action(state)
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.log_probs) == 1
        assert len(agent.values) == 1

    def test_stored_log_prob_is_detached(self, agent):
        """采样时记录的旧 log_prob 应是常数（不带梯度）"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        agent.select_action(state)
        assert not agent.log_probs[0].requires_grad
        assert not agent.values[0].requires_grad

    def test_store_reward(self, agent):
        """store_reward 应正确记录奖励"""
        agent.store_reward(1.0)
        agent.store_reward(0.5)
        assert agent.rewards == [1.0, 0.5]

    def test_update_clears_history(self, agent):
        """update 后应清空所有回合记录"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(5):
            agent.select_action(state)
            agent.store_reward(1.0)
        agent.update()
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.log_probs) == 0
        assert len(agent.values) == 0
        assert len(agent.rewards) == 0

    def test_update_returns_three_losses(self, agent):
        """update 应返回 (total, actor, critic) 三个损失值"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            agent.select_action(state)
            agent.store_reward(1.0)
        result = agent.update()
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    def test_update_empty_returns_zeros(self, agent):
        """没有数据时 update 应返回 (0, 0, 0)"""
        assert agent.update() == (0.0, 0.0, 0.0)

    def test_update_changes_parameters(self, agent):
        """update 应改变网络参数"""
        params_before = [p.clone() for p in agent.net.parameters()]

        state = np.array([0.5, -0.3, 0.1, 0.2], dtype=np.float32)
        for _ in range(20):
            agent.select_action(state)
            agent.store_reward(1.0)
        agent.update()

        changed = any(
            not torch.allclose(b, a)
            for b, a in zip(params_before, agent.net.parameters())
        )
        assert changed

    def test_update_bootstrap(self, agent):
        """带 bootstrap 的 update 应正常工作并返回三个损失"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(5):
            agent.select_action(state)
            agent.store_reward(1.0)
        last_state = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        result = agent.update(last_state=last_state, bootstrap=True)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    def test_anneal_lr(self, agent):
        """anneal_lr 应按进度把学习率从 lr0 线性降到 lr0 的 1/10"""
        agent.anneal_lr(0.0)
        assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(agent.lr0)
        agent.anneal_lr(1.0)
        assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(agent.lr0 * 0.1)


# ===== 训练集成测试 =====


@pytest.mark.slow
class TestTraining:
    """训练流程集成测试（慢速）"""

    def test_train_ppo_runs(self):
        """train_ppo 应能正常运行并返回 5 组等长结果"""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=16,
            update_epochs=3,
            minibatch_size=16,
        )
        rewards, steps, total_losses, actor_losses, critic_losses = train_ppo(
            env, agent, episodes=5, max_steps=50
        )
        env.close()

        assert len(rewards) == 5
        assert len(steps) == 5
        assert len(total_losses) == 5
        assert len(actor_losses) == 5
        assert len(critic_losses) == 5
        assert all(r >= 0 for r in rewards)
        assert all(s >= 1 for s in steps)

    def test_run_ppo_experiment(self):
        """run_ppo_experiment 应返回正确结构的 dict"""
        result = run_ppo_experiment(
            episodes=5,
            hidden_dim=16,
            max_steps=50,
            update_epochs=3,
            minibatch_size=16,
        )

        for key in (
            "episodes",
            "avg_rewards",
            "avg_steps",
            "avg_losses",
            "avg_actor_losses",
            "avg_critic_losses",
            "summary",
        ):
            assert key in result
        assert len(result["avg_rewards"]) == 5
        assert len(result["avg_actor_losses"]) == 5
        assert len(result["avg_critic_losses"]) == 5
        for key in ("final_avg_reward", "max_reward", "solved_episode"):
            assert key in result["summary"]

    def test_callback_called(self):
        """on_episode_end 回调应被正确调用"""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=16,
            update_epochs=2,
            minibatch_size=16,
        )
        call_count = [0]

        def callback(ep, reward, steps, loss):
            call_count[0] += 1

        train_ppo(env, agent, episodes=3, max_steps=20, on_episode_end=callback)
        env.close()

        assert call_count[0] == 3
