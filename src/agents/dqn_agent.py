"""
DQN智能体 - Deep Q-Network算法实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
import random

from .base_agent import BaseAgent
from ..utils.logger import get_logger


class DQNNetwork(nn.Module):
    """DQN网络架构"""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 config: Dict[str, Any]):
        """
        初始化DQN网络
        
        Args:
            observation_space: 观测空间
            action_space: 动作空间
            config: 网络配置
        """
        super(DQNNetwork, self).__init__()
        
        self.config = config
        algo_params = config.get('algorithm_params', {})
        
        # 网络架构配置
        self.net_arch = algo_params.get('net_arch', [512, 512, 256])
        self.activation_fn = self._get_activation_fn(algo_params.get('activation_fn', 'relu'))
        self.use_dueling = algo_params.get('use_dueling', True)
        
        # 处理观测空间
        self.observation_type = self._determine_observation_type(observation_space)
        
        # 动作空间
        if not hasattr(action_space, 'n'):
            raise ValueError("DQN only supports discrete action spaces")
        self.n_actions = action_space.n
        
        if self.observation_type == 'mixed':
            # 混合观测：图像 + 状态
            self.image_encoder = self._build_image_encoder()
            self.state_encoder = self._build_state_encoder(observation_space['state'].shape[0])
            
            # 计算特征维度
            image_features = 512
            state_features = 256
            self.feature_dim = image_features + state_features
            
        elif self.observation_type == 'state':
            # 仅状态观测
            state_dim = observation_space.shape[0]
            self.state_encoder = self._build_state_encoder(state_dim)
            self.feature_dim = 512
            
        else:  # vision only
            # 仅视觉观测
            self.image_encoder = self._build_image_encoder()
            self.feature_dim = 512
        
        # Q网络
        if self.use_dueling:
            self.q_network = self._build_dueling_network()
        else:
            self.q_network = self._build_simple_network()
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _determine_observation_type(self, observation_space) -> str:
        """确定观测类型"""
        if hasattr(observation_space, 'spaces'):
            if 'images' in observation_space.spaces and 'state' in observation_space.spaces:
                return 'mixed'
            elif any(key in observation_space.spaces for key in ['rgb', 'depth', 'segmentation']):
                return 'vision'
        return 'state'
    
    def _get_activation_fn(self, activation_name: str):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        return activations.get(activation_name.lower(), nn.ReLU)
    
    def _build_image_encoder(self) -> nn.Module:
        """构建图像编码器"""
        return nn.Sequential(
            # 处理RGB图像: 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            self.activation_fn(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            self.activation_fn(),
            nn.Dropout(0.1)
        )
    
    def _build_state_encoder(self, state_dim: int) -> nn.Module:
        """构建状态编码器"""
        if self.observation_type == 'mixed':
            # 混合模式下的状态编码器
            return nn.Sequential(
                nn.Linear(state_dim, 128),
                self.activation_fn(),
                nn.Dropout(0.1),
                nn.Linear(128, 256),
                self.activation_fn(),
                nn.Dropout(0.1)
            )
        else:
            # 仅状态模式下的编码器
            layers = []
            input_dim = state_dim
            
            for hidden_dim in [256, 512]:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    self.activation_fn(),
                    nn.Dropout(0.1)
                ])
                input_dim = hidden_dim
            
            return nn.Sequential(*layers)
    
    def _build_simple_network(self) -> nn.Module:
        """构建简单Q网络"""
        layers = []
        input_dim = self.feature_dim
        
        for hidden_dim in self.net_arch:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, self.n_actions))
        
        return nn.Sequential(*layers)
    
    def _build_dueling_network(self) -> nn.Module:
        """构建Dueling DQN网络"""
        # 共享特征层
        shared_layers = []
        input_dim = self.feature_dim
        
        for hidden_dim in self.net_arch[:-1]:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        shared_net = nn.Sequential(*shared_layers)
        
        # 价值流
        value_stream = nn.Sequential(
            nn.Linear(input_dim, self.net_arch[-1]),
            self.activation_fn(),
            nn.Linear(self.net_arch[-1], 1)
        )
        
        # 优势流
        advantage_stream = nn.Sequential(
            nn.Linear(input_dim, self.net_arch[-1]),
            self.activation_fn(),
            nn.Linear(self.net_arch[-1], self.n_actions)
        )
        
        return nn.ModuleDict({
            'shared': shared_net,
            'value': value_stream,
            'advantage': advantage_stream
        })
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode_observations(self, observations):
        """编码观测"""
        if self.observation_type == 'mixed':
            # 处理图像
            images = observations['images']
            rgb_features = images['rgb'] if isinstance(images, dict) else images
            image_encoded = self.image_encoder(rgb_features)
            
            # 处理状态
            state = observations['state']
            state_encoded = self.state_encoder(state)
            
            # 合并特征
            features = torch.cat([image_encoded, state_encoded], dim=-1)
            
        elif self.observation_type == 'state':
            features = self.state_encoder(observations)
            
        else:  # vision
            if isinstance(observations, dict):
                rgb = observations.get('rgb', list(observations.values())[0])
            else:
                rgb = observations
            features = self.image_encoder(rgb)
        
        return features
    
    def forward(self, observations):
        """前向传播"""
        # 编码观测
        features = self.encode_observations(observations)
        
        if self.use_dueling:
            # Dueling DQN
            shared_features = self.q_network['shared'](features)
            
            value = self.q_network['value'](shared_features)
            advantage = self.q_network['advantage'](shared_features)
            
            # 组合价值和优势: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            # 简单DQN
            q_values = self.q_network(features)
        
        return q_values


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数
            beta_start: 重要性采样起始值
            beta_frames: beta增长的帧数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # 存储
        self.buffer = []
        self.pos = 0
        
        # 优先级树
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def beta_by_frame(self, frame_idx):
        """计算当前的beta值"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """采样经验"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # 计算重要性权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta_by_frame(self.frame))
        weights /= weights.max()
        
        # 获取经验
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        """更新优先级"""
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    """简单经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """采样经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """DQN智能体"""
    
    def __init__(self, 
                 env,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化DQN智能体
        
        Args:
            env: 环境实例
            config: 配置字典
            device: 计算设备
        """
        super().__init__(env, "DQN", config, device)
        
        # DQN特定参数
        algo_params = self.config.get('algorithm_params', {})
        self.buffer_size = algo_params.get('buffer_size', 100000)
        self.learning_rate = algo_params.get('learning_rate', 1e-4)
        self.batch_size = algo_params.get('batch_size', 32)
        self.gamma = algo_params.get('gamma', 0.99)
        self.target_update_freq = algo_params.get('target_update_freq', 1000)
        
        # 探索参数
        self.exploration_fraction = algo_params.get('exploration_fraction', 0.3)
        self.exploration_initial_eps = algo_params.get('exploration_initial_eps', 1.0)
        self.exploration_final_eps = algo_params.get('exploration_final_eps', 0.05)
        
        # 训练参数
        self.train_freq = algo_params.get('train_freq', 4)
        self.gradient_steps = algo_params.get('gradient_steps', 1)
        self.learning_starts = algo_params.get('learning_starts', 10000)
        
        # 网络参数
        self.use_double_dqn = algo_params.get('use_double_dqn', True)
        self.use_dueling = algo_params.get('use_dueling', True)
        
        # 优先经验回放
        self.prioritized_replay = algo_params.get('prioritized_replay', True)
        if self.prioritized_replay:
            self.prioritized_replay_alpha = algo_params.get('prioritized_replay_alpha', 0.6)
            self.prioritized_replay_beta0 = algo_params.get('prioritized_replay_beta0', 0.4)
            self.prioritized_replay_beta_iters = algo_params.get('prioritized_replay_beta_iters', 1000000)
        
        # 探索状态
        self.current_eps = self.exploration_initial_eps
        self.update_count = 0
        
        # 构建网络
        self.build_networks()
        
        self.logger.info(f"DQN Agent initialized with buffer_size={self.buffer_size}, "
                        f"double_dqn={self.use_double_dqn}, dueling={self.use_dueling}, "
                        f"prioritized_replay={self.prioritized_replay}")
    
    def build_networks(self):
        """构建神经网络"""
        # 主网络
        self.policy_net = DQNNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        # 目标网络
        self.target_net = DQNNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            eps=1e-4
        )
        
        # 经验回放缓冲区
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                alpha=self.prioritized_replay_alpha,
                beta_start=self.prioritized_replay_beta0,
                beta_frames=self.prioritized_replay_beta_iters
            )
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
    
    def select_action(self, observation, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        if deterministic or (len(self.replay_buffer) < self.learning_starts):
            # 确定性策略或学习开始前
            eps = 0.0 if deterministic else self.current_eps
        else:
            eps = self.current_eps
        
        # ε-贪婪策略
        if np.random.random() < eps:
            # 随机动作
            action = np.random.randint(0, self.action_space.n)
        else:
            # 贪婪动作
            self.policy_net.eval()
            with torch.no_grad():
                obs_tensor = self._obs_to_tensor(observation)
                q_values = self.policy_net(obs_tensor)
                action = q_values.argmax().item()
            self.policy_net.train()
        
        return np.array([action])
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return {}
        
        if self.global_step % self.train_freq != 0:
            return {}
        
        losses = []
        for _ in range(self.gradient_steps):
            loss_info = self._update_step()
            losses.append(loss_info)
        
        # 更新探索率
        self._update_exploration_rate()
        
        # 更新目标网络
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.update_count += 1
        
        # 返回平均损失
        if losses:
            return {
                key: np.mean([loss[key] for loss in losses])
                for key in losses[0].keys()
            }
        return {}
    
    def _update_step(self) -> Dict[str, float]:
        """单步更新"""
        # 采样经验
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # 转换为张量
        states = self._batch_obs_to_tensor(states)
        next_states = self._batch_obs_to_tensor(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态Q值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_q_values = self.policy_net(next_states)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # 标准DQN
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # 目标Q值
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # 计算损失
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        
        self.optimizer.step()
        
        # 更新优先级
        if self.prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy().flatten() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        return {
            'q_loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'exploration_rate': self.current_eps
        }
    
    def _update_exploration_rate(self):
        """更新探索率"""
        if len(self.replay_buffer) < self.learning_starts:
            return
        
        # 线性衰减
        total_steps = self.exploration_fraction * self.total_timesteps
        progress = min(1.0, (self.global_step - self.learning_starts) / total_steps)
        
        self.current_eps = self.exploration_initial_eps + progress * (
            self.exploration_final_eps - self.exploration_initial_eps
        )
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action[0], reward, next_state, done)
    
    def _obs_to_tensor(self, observation):
        """将观测转换为张量"""
        if isinstance(observation, dict):
            obs_dict = {}
            for key, value in observation.items():
                if isinstance(value, dict):
                    obs_dict[key] = {
                        k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                        for k, v in value.items()
                    }
                else:
                    obs_dict[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            return obs_dict
        else:
            return torch.FloatTensor(observation).unsqueeze(0).to(self.device)
    
    def _batch_obs_to_tensor(self, observations):
        """将观测批次转换为张量"""
        if isinstance(observations[0], dict):
            obs_dict = {}
            for key in observations[0].keys():
                if isinstance(observations[0][key], dict):
                    obs_dict[key] = {}
                    for sub_key in observations[0][key].keys():
                        obs_dict[key][sub_key] = torch.FloatTensor(
                            [obs[key][sub_key] for obs in observations]
                        ).to(self.device)
                else:
                    obs_dict[key] = torch.FloatTensor(
                        [obs[key] for obs in observations]
                    ).to(self.device)
            return obs_dict
        else:
            return torch.FloatTensor(observations).to(self.device)
    
    def save_model(self, filepath: str, episode: int):
        """保存模型"""
        checkpoint = {
            'episode': episode,
            'global_step': self.global_step,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'algorithm': self.algorithm_name,
            'current_eps': self.current_eps,
            'update_count': self.update_count
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.current_eps = checkpoint.get('current_eps', self.exploration_final_eps)
        self.update_count = checkpoint.get('update_count', 0)
        
        self.logger.info(f"DQN model loaded from {filepath}")
    
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """训练单个回合（重写基类方法）"""
        observation, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # 选择动作
            action = self.select_action(observation, deterministic=False)
            
            # 执行动作
            next_observation, reward, done, truncated, info = self.env.step(action)
            
            # 存储经验
            self.store_transition(observation, action, reward, next_observation, done or truncated)
            
            # 更新网络
            if self.global_step > self.learning_starts:
                update_info = self.update()
                if update_info and self.global_step % self.log_interval == 0:
                    self._log_update_info(update_info)
            
            # 更新状态
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
        
        # 回合结束处理
        success = info.get('success', False)
        if success:
            self.logger.info(f"Episode {episode} SUCCESS - Reward: {episode_reward:.2f}, Length: {episode_length}")
        
        return {
            'episode': episode,
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'success': success
        }