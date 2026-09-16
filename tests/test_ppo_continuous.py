"""连续动作版 PPO（高斯策略）模块单元测试"""

import numpy as np
import pytest
import torch
import gymnasium as gym

from core.ppo_continuous import (
    GaussianActorCriticNetwork,
    PPOContinuousAgent,
    train_ppo_continuous,
    run_ppo_continuous_experiment,
)


# ===== GaussianActorCriticNetwork 测试 =====


class TestGaussianNetwork:
    """高斯策略网络结构测试"""

    @pytest.fixture
    def net(self):
        return GaussianActorCriticNetwork(state_dim=27, action_dim=8, hidden_dim=32)

    def test_forward_shapes(self, net):
        """forward 应返回 ([B,8] 均值, [8] 标准差, [B,1] 价值)"""
        x = torch.randn(5, 27)
        mean, std, value = net(x)
        assert mean.shape == (5, 8)
        assert std.shape == (8,)
        assert value.shape == (5, 1)

    def test_std_is_positive(self, net):
        """标准差 = exp(log_std)，必须恒为正"""
        _, std, _ = net(torch.randn(1, 27))
        assert (std > 0).all()

    def test_log_std_is_trainable(self, net):
        """log_std 应是可学习参数"""
        assert net.log_std.requires_grad


# ===== PPOContinuousAgent 测试 =====


class TestPPOContinuousAgent:
    """连续动作 PPO 智能体测试"""

    @pytest.fixture
    def agent(self):
        return PPOContinuousAgent(
            state_dim=4,
            action_dim=2,
            action_low=np.array([-1.0, -1.0]),
            action_high=np.array([1.0, 1.0]),
            hidden_dim=32,
            lr=3e-4,
            update_epochs=3,
            minibatch_size=16,
        )

    def test_select_action_shape_and_type(self, agent):
        """选出的动作应是 action_dim 维的 np 数组"""
        state = np.array([0.0, 0.1, -0.05, 0.2], dtype=np.float32)
        action = agent.select_action(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)

    def test_select_action_clipped_to_bounds(self, agent):
        """动作应被裁剪到 [action_low, action_high] 范围内"""
        state = np.array([0.0, 0.1, -0.05, 0.2], dtype=np.float32)
        for _ in range(50):
            action = agent.select_action(state)
            assert (action >= -1.0).all()
            assert (action <= 1.0).all()

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

    def test_store_reward_records_episode_end(self, agent):
        """store_reward 应记录回合结束标记和自举价值"""
        agent.store_reward(1.0)
        agent.store_reward(0.5, episode_end=True, end_value=0.0)
        agent.store_reward(0.5, episode_end=True, end_value=2.5)
        assert agent.rewards == [1.0, 0.5, 0.5]
        assert agent.episode_ends == [False, True, True]
        assert agent.end_values == [0.0, 0.0, 2.5]

    def test_update_clears_history(self, agent):
        """update 后应清空所有批次记录"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(5):
            agent.select_action(state)
            agent.store_reward(1.0, episode_end=(i == 4), end_value=0.0)
        agent.update()
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.log_probs) == 0
        assert len(agent.values) == 0
        assert len(agent.rewards) == 0
        assert len(agent.episode_ends) == 0
        assert len(agent.end_values) == 0

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
        """update 应改变网络参数（含 log_std）"""
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

    def test_update_cross_episode_batch(self, agent):
        """一批数据跨多个回合（含 terminated/truncated/中途截断）应正常更新"""
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            agent.select_action(state)
            agent.store_reward(1.0)
        # 第 10 步 terminated（末状态价值算 0）
        agent.select_action(state)
        agent.store_reward(1.0, episode_end=True, end_value=0.0)
        for _ in range(9):
            agent.select_action(state)
            agent.store_reward(1.0)
        # 第 20 步 truncated（用 V(末状态) 自举，采样时已算好传入）
        agent.select_action(state)
        agent.store_reward(1.0, episode_end=True, end_value=0.5)
        # 批次末尾停在回合中途：bootstrap=True
        result = agent.update(last_state=state, bootstrap=True)
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
class TestTrainingContinuous:
    """训练流程集成测试（慢速，用最小 MuJoCo 环境 InvertedPendulum）"""

    def test_train_ppo_continuous_runs(self):
        """按步数攒批训练应返回 5 组等长结果（长度 = 更新次数）"""
        env = gym.make("InvertedPendulum-v5")
        agent = PPOContinuousAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dim=16,
            update_epochs=3,
            minibatch_size=16,
        )
        # 250 步 / 每批 50 步 = 5 次更新
        rewards, steps, total_losses, actor_losses, critic_losses = (
            train_ppo_continuous(env, agent, total_steps=250, batch_size=50)
        )
        env.close()

        assert len(rewards) == 5
        assert len(steps) == 5
        assert len(total_losses) == 5
        assert len(actor_losses) == 5
        assert len(critic_losses) == 5
        assert all(r >= 0 for r in rewards)

    def test_train_stops_at_episode_boundary_correctly(self):
        """跨回合采样：环境应在回合结束后自动重置，采样不间断"""
        env = gym.make("InvertedPendulum-v5")
        agent = PPOContinuousAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=16,
            update_epochs=2,
            minibatch_size=16,
        )
        # 采 400 步（跨回合也没关系，不应报错）
        train_ppo_continuous(env, agent, total_steps=400, batch_size=100)
        env.close()

    def test_run_ppo_continuous_experiment(self):
        """run_ppo_continuous_experiment 应返回正确结构的 dict"""
        result = run_ppo_continuous_experiment(
            env_id="InvertedPendulum-v5",
            total_steps=250,
            batch_size=50,
            hidden_dim=16,
            update_epochs=3,
            minibatch_size=16,
        )

        for key in (
            "env_id",
            "total_steps",
            "avg_rewards",
            "avg_steps",
            "avg_losses",
            "avg_actor_losses",
            "avg_critic_losses",
            "summary",
        ):
            assert key in result
        assert result["env_id"] == "InvertedPendulum-v5"
        assert len(result["avg_rewards"]) == 5
        for key in ("final_avg_reward", "initial_avg_reward", "max_reward"):
            assert key in result["summary"]

    def test_callback_called(self):
        """on_update_end 回调应按更新次数调用"""
        env = gym.make("InvertedPendulum-v5")
        agent = PPOContinuousAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dim=16,
            update_epochs=2,
            minibatch_size=16,
        )
        call_count = [0]

        def callback(idx, reward, steps, loss, global_step):
            call_count[0] += 1

        train_ppo_continuous(
            env, agent, total_steps=150, batch_size=50, on_update_end=callback
        )
        env.close()

        assert call_count[0] == 3
