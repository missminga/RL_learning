"""Policy Gradient (REINFORCE) 模块单元测试"""

import numpy as np
import pytest
import torch
import gymnasium as gym

from core.policy_gradient import (
    PolicyNetwork,
    REINFORCEAgent,
    train_reinforce,
    run_reinforce_experiment,
)


# ===== PolicyNetwork 测试 =====


class TestPolicyNetwork:
    """策略网络基础测试"""

    def test_output_shape(self):
        """输出维度应等于动作数"""
        net = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_batch_output(self):
        """批量输入应返回对应批量输出"""
        net = PolicyNetwork(state_dim=4, action_dim=3, hidden_dim=16)
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 3)

    def test_action_probs_sum_to_one(self):
        """动作概率之和应为 1"""
        net = PolicyNetwork(state_dim=4, action_dim=3, hidden_dim=32)
        x = torch.randn(1, 4)
        probs = net.get_action_probs(x)
        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)

    def test_action_probs_non_negative(self):
        """动作概率应为非负"""
        net = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        x = torch.randn(5, 4)
        probs = net.get_action_probs(x)
        assert (probs >= 0).all()

    def test_different_inputs_different_outputs(self):
        """不同输入应产生不同输出"""
        net = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        x1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        x2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        out1 = net(x1)
        out2 = net(x2)
        assert not torch.allclose(out1, out2)


# ===== REINFORCEAgent 测试 =====


class TestREINFORCEAgent:
    """REINFORCE 智能体测试"""

    @pytest.fixture
    def agent(self):
        return REINFORCEAgent(
            state_dim=4, action_dim=2, hidden_dim=32,
            lr=1e-3, gamma=0.99,
        )

    def test_select_action_range(self, agent):
        """选出的动作应在有效范围内"""
        state = np.array([0.0, 0.1, -0.05, 0.2], dtype=np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert action in [0, 1]
        # 清空记录
        agent.log_probs = []
        agent.rewards = []

    def test_select_action_records_log_prob(self, agent):
        """select_action 应记录 log_prob"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert len(agent.log_probs) == 0
        agent.select_action(state)
        assert len(agent.log_probs) == 1
        agent.select_action(state)
        assert len(agent.log_probs) == 2
        agent.log_probs = []
        agent.rewards = []

    def test_store_reward(self, agent):
        """store_reward 应正确记录奖励"""
        agent.store_reward(1.0)
        agent.store_reward(0.5)
        assert len(agent.rewards) == 2
        assert agent.rewards == [1.0, 0.5]

    def test_update_clears_history(self, agent):
        """update 后应清空回合记录"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(5):
            agent.select_action(state)
            agent.store_reward(1.0)
        agent.update()
        assert len(agent.log_probs) == 0
        assert len(agent.rewards) == 0

    def test_update_returns_loss(self, agent):
        """update 应返回损失值"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            agent.select_action(state)
            agent.store_reward(1.0)
        loss = agent.update()
        assert isinstance(loss, float)

    def test_update_empty_returns_zero(self, agent):
        """没有数据时 update 应返回 0"""
        loss = agent.update()
        assert loss == 0.0

    def test_update_changes_parameters(self, agent):
        """update 应改变网络参数"""
        # 记录更新前的参数
        params_before = [p.clone() for p in agent.policy_net.parameters()]

        state = np.array([0.5, -0.3, 0.1, 0.2], dtype=np.float32)
        for _ in range(20):
            agent.select_action(state)
            agent.store_reward(1.0)
        agent.update()

        # 至少有一个参数应该变了
        changed = False
        for p_before, p_after in zip(params_before, agent.policy_net.parameters()):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed


# ===== 训练集成测试 =====


class TestTraining:
    """训练流程集成测试"""

    def test_train_reinforce_runs(self):
        """train_reinforce 应能正常运行并返回正确长度的结果"""
        env = gym.make("CartPole-v1")
        agent = REINFORCEAgent(
            state_dim=4, action_dim=2, hidden_dim=16,
            lr=1e-3, gamma=0.99,
        )
        rewards, steps, losses = train_reinforce(
            env, agent, episodes=5, max_steps=50,
        )
        env.close()

        assert len(rewards) == 5
        assert len(steps) == 5
        assert len(losses) == 5
        assert all(r >= 0 for r in rewards)
        assert all(s >= 1 for s in steps)

    def test_run_reinforce_experiment(self):
        """run_reinforce_experiment 应返回正确结构的 dict"""
        result = run_reinforce_experiment(
            episodes=5, hidden_dim=16, max_steps=50,
        )

        assert "episodes" in result
        assert "avg_rewards" in result
        assert "avg_steps" in result
        assert "avg_losses" in result
        assert "summary" in result
        assert len(result["avg_rewards"]) == 5
        assert "final_avg_reward" in result["summary"]
        assert "max_reward" in result["summary"]
        assert "solved_episode" in result["summary"]

    def test_callback_called(self):
        """on_episode_end 回调应被正确调用"""
        env = gym.make("CartPole-v1")
        agent = REINFORCEAgent(
            state_dim=4, action_dim=2, hidden_dim=16,
        )
        call_count = [0]

        def callback(ep, reward, steps, loss):
            call_count[0] += 1

        train_reinforce(env, agent, episodes=3, max_steps=20,
                        on_episode_end=callback)
        env.close()

        assert call_count[0] == 3
