"""
SAC智能体 - Soft Actor-Critic算法实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
import random

from .base_agent import BaseAgent
from ..utils.logger import get_logger


class SACNetwork(nn.Module):
    """SAC网络基类"""
    
    def __init__(self, 
                 observation_space,
                 config: Dict[str, Any]):
        """
        初始化网络基类
        
        Args:
            observation_space: 观测空间
            config: 网络配置
        """
        super(SACNetwork, self).__init__()
        
        self.config = config
        policy_config = config.get('policy_kwargs', {})
        
        # 网络架构配置
        self.net_arch = policy_config.get('net_arch', [256, 256])
        self.activation_fn = self._get_activation_fn(policy_config.get('activation_fn', 'relu'))
        
        # 处理观测空间
        self.observation_type = self._determine_observation_type(observation_space)
        
        if self.observation_type == 'mixed':
            # 混合观测：图像 + 状态
            self.image_encoder = self._build_image_encoder()
            self.state_encoder = self._build_state_encoder(observation_space['state'].shape[0])
            
            # 计算特征维度
            image_features = 512
            state_features = 128
            self.feature_dim = image_features + state_features
            
        elif self.observation_type == 'state':
            # 仅状态观测
            state_dim = observation_space.shape[0]
            self.state_encoder = self._build_state_encoder(state_dim)
            self.feature_dim = 256
            
        else:  # vision only
            # 仅视觉观测
            self.image_encoder = self._build_image_encoder()
            self.feature_dim = 512
    
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
                nn.Linear(state_dim, 64),
                self.activation_fn(),
                nn.Linear(64, 128),
                self.activation_fn()
            )
        else:
            # 仅状态模式下的编码器
            return nn.Sequential(
                nn.Linear(state_dim, 128),
                self.activation_fn(),
                nn.Linear(128, 256),
                self.activation_fn()
            )
    
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


class SACPolicyNetwork(SACNetwork):
    """SAC策略网络"""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 config: Dict[str, Any]):
        """
        初始化SAC策略网络
        
        Args:
            observation_space: 观测空间
            action_space: 动作空间
            config: 配置
        """
        super(SACPolicyNetwork, self).__init__(observation_space, config)
        
        # 动作空间
        if hasattr(action_space, 'n'):
            raise ValueError("SAC only supports continuous action spaces")
        self.action_dim = action_space.shape[0]
        
        # 策略网络参数
        policy_config = config.get('policy_kwargs', {})
        self.log_std_init = policy_config.get('log_std_init', -3)
        self.log_std_min = -20
        self.log_std_max = 2
        
        # 构建网络
        self.shared_net = self._build_shared_layers()
        self.mean_layer = nn.Linear(self.net_arch[-1], self.action_dim)
        self.log_std_layer = nn.Linear(self.net_arch[-1], self.action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _build_shared_layers(self) -> nn.Module:
        """构建共享层"""
        layers = []
        input_dim = self.feature_dim
        
        for hidden_dim in self.net_arch:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn()
            ])
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations):
        """前向传播"""
        # 编码观测
        features = self.encode_observations(observations)
        
        # 共享层
        shared_features = self.shared_net(features)
        
        # 均值和对数标准差
        mean = self.mean_layer(shared_features)
        log_std = self.log_std_layer(shared_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, observations):
        """采样动作"""
        mean, log_std = self.forward(observations)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 可反向传播的采样
        
        # tanh压缩到(-1, 1)
        action = torch.tanh(x_t)
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的雅可比行列式
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class SACQNetwork(SACNetwork):
    """SAC Q网络"""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 config: Dict[str, Any]):
        """
        初始化SAC Q网络
        
        Args:
            observation_space: 观测空间
            action_space: 动作空间
            config: 配置
        """
        super(SACQNetwork, self).__init__(observation_space, config)
        
        # 动作空间
        self.action_dim = action_space.shape[0]
        
        # 构建Q网络
        self.q_network = self._build_q_network()
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _build_q_network(self) -> nn.Module:
        """构建Q网络"""
        layers = []
        input_dim = self.feature_dim + self.action_dim  # 观测特征 + 动作
        
        for hidden_dim in self.net_arch:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn()
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations, actions):
        """前向传播"""
        # 编码观测
        obs_features = self.encode_observations(observations)
        
        # 合并观测特征和动作
        q_input = torch.cat([obs_features, actions], dim=-1)
        
        # Q值
        q_value = self.q_network(q_input)
        
        return q_value


class SACReplayBuffer:
    """SAC经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
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


class SACAgent(BaseAgent):
    """SAC智能体"""
    
    def __init__(self, 
                 env,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化SAC智能体
        
        Args:
            env: 环境实例
            config: 配置字典
            device: 计算设备
        """
        super().__init__(env, "SAC", config, device)
        
        # SAC特定参数
        algo_params = self.config.get('algorithm_params', {})
        self.learning_rate = algo_params.get('learning_rate', 3e-4)
        self.buffer_size = algo_params.get('buffer_size', 1000000)
        self.batch_size = algo_params.get('batch_size', 256)
        self.gamma = algo_params.get('gamma', 0.99)
        self.tau = algo_params.get('tau', 0.005)
        
        # 熵参数
        self.ent_coef = algo_params.get('ent_coef', 'auto')
        self.target_entropy = algo_params.get('target_entropy', 'auto')
        if self.target_entropy == 'auto':
            self.target_entropy = -float(self.action_space.shape[0])
        self.ent_coef_lr = algo_params.get('ent_coef_lr', 3e-4)
        
        # 训练参数
        self.train_freq = algo_params.get('train_freq', 1)
        self.gradient_steps = algo_params.get('gradient_steps', 1)
        self.learning_starts = algo_params.get('learning_starts', 10000)
        self.target_update_interval = algo_params.get('target_update_interval', 1)
        
        # 网络更新计数
        self.update_count = 0
        
        # 构建网络
        self.build_networks()
        
        self.logger.info(f"SAC Agent initialized with buffer_size={self.buffer_size}, "
                        f"ent_coef={self.ent_coef}, target_entropy={self.target_entropy}")
    
    def build_networks(self):
        """构建神经网络"""
        # 策略网络
        self.policy_net = SACPolicyNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        # Q网络（双Q学习）
        self.q_net1 = SACQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        self.q_net2 = SACQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        # 目标Q网络
        self.target_q_net1 = SACQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        self.target_q_net2 = SACQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        self.q1_optimizer = optim.Adam(
            self.q_net1.parameters(),
            lr=self.learning_rate
        )
        
        self.q2_optimizer = optim.Adam(
            self.q_net2.parameters(),
            lr=self.learning_rate
        )
        
        # 熵系数
        if self.ent_coef == 'auto':
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
            self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=self.ent_coef_lr)
        else:
            self.log_ent_coef = torch.log(torch.FloatTensor([self.ent_coef])).to(self.device)
        
        # 经验回放缓冲区
        self.replay_buffer = SACReplayBuffer(self.buffer_size)
    
    def select_action(self, observation, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        self.policy_net.eval()
        
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(observation)
            
            if deterministic:
                # 确定性策略：使用均值
                mean, _ = self.policy_net.forward(obs_tensor)
                action = torch.tanh(mean)
            else:
                # 随机策略：采样
                action, _ = self.policy_net.sample(obs_tensor)
        
        self.policy_net.train()
        
        # 转换到动作空间范围
        action = action.cpu().numpy()[0]
        
        # 缩放到实际动作范围
        action = self._scale_action(action)
        
        return action
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """将动作从[-1,1]缩放到实际范围"""
        # 获取动作边界
        env_config = self.config.get('env_config', {})
        action_bounds = env_config.get('action_bounds', {
            'velocity_x': [-5.0, 5.0],
            'velocity_y': [-5.0, 5.0],
            'velocity_z': [-2.0, 2.0],
            'yaw_rate': [-90.0, 90.0]
        })
        
        # 缩放每个动作分量
        scaled_action = np.zeros_like(action)
        bounds_list = [
            action_bounds['velocity_x'],
            action_bounds['velocity_y'],
            action_bounds['velocity_z'],
            action_bounds['yaw_rate']
        ]
        
        for i in range(len(action)):
            low, high = bounds_list[i]
            scaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)
        
        return scaled_action
    
    def _unscale_action(self, action: np.ndarray) -> np.ndarray:
        """将动作从实际范围缩放到[-1,1]"""
        env_config = self.config.get('env_config', {})
        action_bounds = env_config.get('action_bounds', {
            'velocity_x': [-5.0, 5.0],
            'velocity_y': [-5.0, 5.0],
            'velocity_z': [-2.0, 2.0],
            'yaw_rate': [-90.0, 90.0]
        })
        
        unscaled_action = np.zeros_like(action)
        bounds_list = [
            action_bounds['velocity_x'],
            action_bounds['velocity_y'],
            action_bounds['velocity_z'],
            action_bounds['yaw_rate']
        ]
        
        for i in range(len(action)):
            low, high = bounds_list[i]
            unscaled_action[i] = 2.0 * (action[i] - low) / (high - low) - 1.0
        
        return unscaled_action
    
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
        
        # 软更新目标网络
        if self.update_count % self.target_update_interval == 0:
            self._soft_update_target_networks()
        
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
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = self._batch_obs_to_tensor(states)
        next_states = self._batch_obs_to_tensor(next_states)
        
        # 取消缩放动作
        actions = torch.FloatTensor([self._unscale_action(a) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Q网络
        q1_loss, q2_loss = self._update_q_networks(states, actions, rewards, next_states, dones)
        
        # 更新策略网络
        policy_loss = self._update_policy_network(states)
        
        # 更新熵系数
        ent_coef_loss = 0.0
        if self.ent_coef == 'auto':
            ent_coef_loss = self._update_entropy_coefficient(states)
        
        return {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'ent_coef_loss': ent_coef_loss,
            'ent_coef': torch.exp(self.log_ent_coef).item()
        }
    
    def _update_q_networks(self, states, actions, rewards, next_states, dones):
        """更新Q网络"""
        # 当前Q值
        current_q1 = self.q_net1(states, actions)
        current_q2 = self.q_net2(states, actions)
        
        # 目标Q值
        with torch.no_grad():
            # 下一状态的动作和对数概率
            next_actions, next_log_probs = self.policy_net.sample(next_states)
            
            # 目标Q值（双Q学习，取最小值）
            target_q1 = self.target_q_net1(next_states, next_actions)
            target_q2 = self.target_q_net2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # 熵正则化
            ent_coef = torch.exp(self.log_ent_coef)
            target_q = target_q - ent_coef * next_log_probs
            
            # 贝尔曼更新
            target_q = rewards + (self.gamma * target_q * (~dones))
        
        # Q损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新Q网络
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        return q1_loss.item(), q2_loss.item()
    
    def _update_policy_network(self, states):
        """更新策略网络"""
        # 当前策略的动作和对数概率
        actions, log_probs = self.policy_net.sample(states)
        
        # Q值（双Q学习，取最小值）
        q1_values = self.q_net1(states, actions)
        q2_values = self.q_net2(states, actions)
        q_values = torch.min(q1_values, q2_values)
        
        # 策略损失（最大化Q值和熵）
        ent_coef = torch.exp(self.log_ent_coef).detach()
        policy_loss = (ent_coef * log_probs - q_values).mean()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def _update_entropy_coefficient(self, states):
        """更新熵系数"""
        # 当前策略的对数概率
        with torch.no_grad():
            _, log_probs = self.policy_net.sample(states)
        
        # 熵系数损失
        ent_coef_loss = -(self.log_ent_coef * (log_probs + self.target_entropy)).mean()
        
        # 更新熵系数
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        
        return ent_coef_loss.item()
    
    def _soft_update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
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
            'q_net1_state_dict': self.q_net1.state_dict(),
            'q_net2_state_dict': self.q_net2.state_dict(),
            'target_q_net1_state_dict': self.target_q_net1.state_dict(),
            'target_q_net2_state_dict': self.target_q_net2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'log_ent_coef': self.log_ent_coef,
            'config': self.config,
            'algorithm': self.algorithm_name,
            'update_count': self.update_count
        }
        
        if self.ent_coef == 'auto':
            checkpoint['ent_coef_optimizer_state_dict'] = self.ent_coef_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"SAC model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.q_net1.load_state_dict(checkpoint['q_net1_state_dict'])
        self.q_net2.load_state_dict(checkpoint['q_net2_state_dict'])
        self.target_q_net1.load_state_dict(checkpoint['target_q_net1_state_dict'])
        self.target_q_net2.load_state_dict(checkpoint['target_q_net2_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        self.log_ent_coef = checkpoint['log_ent_coef']
        if self.ent_coef == 'auto' and 'ent_coef_optimizer_state_dict' in checkpoint:
            self.ent_coef_optimizer.load_state_dict(checkpoint['ent_coef_optimizer_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.update_count = checkpoint.get('update_count', 0)
        
        self.logger.info(f"SAC model loaded from {filepath}")
    
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