"""
PPO智能体 - Proximal Policy Optimization算法实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
import random

from .base_agent import BaseAgent
from ..utils.logger import get_logger


class PPONetwork(nn.Module):
    """PPO网络（Actor-Critic架构）"""
    
    def __init__(self, 
                 observation_space,
                 action_space, 
                 config: Dict[str, Any]):
        """
        初始化PPO网络
        
        Args:
            observation_space: 观测空间
            action_space: 动作空间
            config: 网络配置
        """
        super(PPONetwork, self).__init__()
        
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
            image_features = 512 * 3  # RGB, depth, segmentation 各512维
            state_features = observation_space['state'].shape[0]  # 状态维度
            self.feature_dim = image_features + state_features
            
        elif self.observation_type == 'state':
            # 仅状态观测
            state_dim = observation_space.shape[0]
            self.state_encoder = self._build_state_encoder(state_dim)
            self.feature_dim = self.net_arch[0]
            
        else:  # vision only
            # 仅视觉观测
            self.image_encoder = self._build_image_encoder()
            self.feature_dim = 512
        
        # 共享网络层
        self.shared_layers = self._build_shared_layers()
        
        # Actor网络（策略）
        self.actor = self._build_actor(action_space)
        
        # Critic网络（价值函数）
        self.critic = self._build_critic()
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _determine_observation_type(self, observation_space) -> str:
        """确定观测类型"""
        if hasattr(observation_space, 'spaces'):
            if 'images' in observation_space.spaces and 'state' in observation_space.spaces:
                return 'mixed'
            elif 'rgb' in observation_space.spaces:
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
        # 创建一个模块字典来处理不同通道数的图像
        return nn.ModuleDict({
            'rgb_encoder': self._build_cnn_encoder(3),      # RGB: 3 channels
            'depth_encoder': self._build_cnn_encoder(1),    # Depth: 1 channel  
            'seg_encoder': self._build_cnn_encoder(3)       # Segmentation: 3 channels
        })
    
    def _build_cnn_encoder(self, in_channels: int) -> nn.Module:
        """构建CNN编码器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
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
        layers = []
        input_dim = state_dim
        
        for hidden_dim in self.net_arch[:-1]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_shared_layers(self) -> nn.Module:
        """构建共享网络层"""
        layers = []
        input_dim = self.feature_dim
        
        for hidden_dim in self.net_arch:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_actor(self, action_space) -> nn.Module:
        """构建Actor网络"""
        if hasattr(action_space, 'n'):
            # 离散动作空间
            return nn.Sequential(
                nn.Linear(self.net_arch[-1], action_space.n),
                nn.Softmax(dim=-1)
            )
        else:
            # 连续动作空间
            action_dim = action_space.shape[0]
            
            # 均值网络
            mean_net = nn.Sequential(
                nn.Linear(self.net_arch[-1], action_dim),
                nn.Tanh()  # 输出范围[-1, 1]
            )
            
            # 标准差参数
            log_std = nn.Parameter(torch.zeros(action_dim))
            
            return nn.ModuleDict({
                'mean': mean_net,
                'log_std': nn.ParameterList([log_std])
            })
    
    def _build_critic(self) -> nn.Module:
        """构建Critic网络"""
        return nn.Linear(self.net_arch[-1], 1)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode_observations(self, observations):
        """编码观测"""
        if self.observation_type == 'mixed':
            # 处理图像
            images = observations['images']
            
            # 处理所有图像类型
            image_features = []
            encoder_map = {'rgb': 'rgb_encoder', 'depth': 'depth_encoder', 'segmentation': 'seg_encoder'}
            
            for img_type in ['rgb', 'depth', 'segmentation']:
                if img_type in images:
                    encoder_name = encoder_map[img_type]
                    img_feat = self.image_encoder[encoder_name](images[img_type])
                    image_features.append(img_feat)
            
            # 合并所有图像特征
            image_encoded = torch.cat(image_features, dim=-1) if image_features else torch.zeros((observations['state'].shape[0], 512*3), device=observations['state'].device)
            
            # 处理状态 (直接使用状态向量)
            state = observations['state']
            state_encoded = state
            
            # 合并特征
            features = torch.cat([image_encoded, state_encoded], dim=-1)
            
        elif self.observation_type == 'state':
            features = self.state_encoder(observations)
            
        else:  # vision
            if isinstance(observations, dict):
                rgb = observations.get('rgb', list(observations.values())[0])
            else:
                rgb = observations
            features = self.image_encoder['rgb_encoder'](rgb)
        
        return features
    
    def forward(self, observations):
        """前向传播"""
        # 编码观测
        features = self.encode_observations(observations)
        
        # 共享层
        shared_features = self.shared_layers(features)
        
        # Actor输出
        if isinstance(self.actor, nn.ModuleDict):
            # 连续动作
            action_mean = self.actor['mean'](shared_features)
            log_std = self.actor['log_std'][0]
            action_std = torch.exp(log_std)
            action_dist = Normal(action_mean, action_std)
        else:
            # 离散动作
            action_probs = self.actor(shared_features)
            action_dist = Categorical(action_probs)
        
        # Critic输出
        state_value = self.critic(shared_features)
        
        return action_dist, state_value.squeeze(-1)


class PPOAgent(BaseAgent):
    """PPO智能体"""
    
    def __init__(self, 
                 env,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化PPO智能体
        
        Args:
            env: 环境实例
            config: 配置字典
            device: 计算设备
        """
        super().__init__(env, "PPO", config, device)
        
        # PPO特定参数
        algo_params = self.config.get('algorithm_params', {})
        self.clip_epsilon = algo_params.get('clip_epsilon', 0.2)
        self.value_loss_coef = algo_params.get('value_loss_coef', 0.5)
        self.entropy_coef = algo_params.get('entropy_coef', 0.01)
        self.max_grad_norm = algo_params.get('max_grad_norm', 0.5)
        self.ppo_epochs = algo_params.get('ppo_epochs', 4)
        self.mini_batch_size = algo_params.get('mini_batch_size', 64)
        self.n_steps = algo_params.get('n_steps', 2048)
        
        # 学习率和优化器参数
        self.learning_rate = algo_params.get('learning_rate', 3e-4)
        self.gamma = algo_params.get('gamma', 0.99)
        self.gae_lambda = algo_params.get('gae_lambda', 0.95)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device
        )
        
        # 构建网络
        self.build_networks()
        
        self.logger.info(f"PPO Agent initialized with clip_epsilon={self.clip_epsilon}, "
                        f"learning_rate={self.learning_rate}")
    
    def build_networks(self):
        """构建神经网络"""
        self.policy_net = PPONetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.config
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # 学习率调度器
        lr_schedule = self.config.get('algorithm_params', {}).get('lr_schedule', 'constant')
        if lr_schedule == 'linear':
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.total_timesteps // self.n_steps
            )
        else:
            self.lr_scheduler = None
    
    def select_action(self, observation, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        self.policy_net.eval()
        
        with torch.no_grad():
            # 转换观测为张量
            obs_tensor = self._obs_to_tensor(observation)
            
            # 获取动作分布和状态值
            action_dist, state_value = self.policy_net(obs_tensor)
            
            # 选择动作
            if deterministic:
                if isinstance(action_dist, Normal):
                    action = action_dist.mean
                else:  # Categorical
                    action = torch.argmax(action_dist.probs, dim=-1)
            else:
                action = action_dist.sample()
            
            # 计算对数概率
            log_prob = action_dist.log_prob(action).sum(dim=-1) if isinstance(action_dist, Normal) else action_dist.log_prob(action)
        
        # 存储信息到缓冲区
        if not deterministic:
            self.buffer.store_step(
                observation=observation,
                action=action.cpu().numpy(),
                log_prob=log_prob.cpu().numpy(),
                value=state_value.cpu().numpy()
            )
        
        self.policy_net.train()
        
        return action.cpu().numpy()
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        """更新网络参数"""
        # 如果缓冲区未满，不进行更新
        if not self.buffer.is_full():
            return {}
        
        # 计算优势和回报
        self.buffer.compute_advantages_and_returns(self.gamma, self.gae_lambda)
        
        # 获取批次数据
        batch_data = self.buffer.get_batch()
        
        # 进行多轮更新
        total_losses = []
        for epoch in range(self.ppo_epochs):
            epoch_losses = []
            
            # 随机打乱数据
            indices = torch.randperm(batch_data['observations'].shape[0])
            
            # 分批更新
            for start_idx in range(0, len(indices), self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                batch_indices = indices[start_idx:end_idx]
                
                mini_batch = {key: value[batch_indices] for key, value in batch_data.items()}
                
                loss_info = self._update_mini_batch(mini_batch)
                epoch_losses.append(loss_info)
            
            # 计算平均损失
            if epoch_losses:
                epoch_avg_loss = {
                    key: np.mean([loss[key] for loss in epoch_losses])
                    for key in epoch_losses[0].keys()
                }
                total_losses.append(epoch_avg_loss)
        
        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 返回平均损失
        if total_losses:
            return {
                key: np.mean([loss[key] for loss in total_losses])
                for key in total_losses[0].keys()
            }
        
        return {}
    
    def _update_mini_batch(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新小批次"""
        # 获取当前策略输出
        action_dist, state_values = self.policy_net(mini_batch['observations'])
        
        # 计算新的对数概率
        if isinstance(action_dist, Normal):
            new_log_probs = action_dist.log_prob(mini_batch['actions']).sum(dim=-1)
        else:  # Categorical
            new_log_probs = action_dist.log_prob(mini_batch['actions'])
        
        # 计算比率
        ratio = torch.exp(new_log_probs - mini_batch['old_log_probs'])
        
        # 计算代理损失
        advantages = mini_batch['advantages']
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        value_loss = F.mse_loss(state_values, mini_batch['returns'])
        
        # 熵损失（鼓励探索）
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # 返回损失信息
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'approx_kl': ((ratio - 1) - (new_log_probs - mini_batch['old_log_probs'])).mean().item(),
            'clip_fraction': ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
        }
    
    def _obs_to_tensor(self, observation):
        """将观测转换为张量"""
        if isinstance(observation, dict):
            obs_dict = {}
            for key, value in observation.items():
                if isinstance(value, dict):
                    # 嵌套字典（如images）
                    obs_dict[key] = {
                        k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                        for k, v in value.items()
                    }
                else:
                    obs_dict[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            return obs_dict
        else:
            return torch.FloatTensor(observation).unsqueeze(0).to(self.device)
    
    def _should_update(self) -> bool:
        """判断是否应该更新"""
        return self.buffer.is_full()
    
    def save_model(self, filepath: str, episode: int):
        """保存模型"""
        checkpoint = {
            'episode': episode,
            'global_step': self.global_step,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'algorithm': self.algorithm_name
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"PPO model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        
        self.logger.info(f"PPO model loaded from {filepath}")


class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self, buffer_size: int, observation_space, action_space, device):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            observation_space: 观测空间
            action_space: 动作空间
            device: 计算设备
        """
        self.buffer_size = buffer_size
        self.device = device
        
        # 存储数组
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # GAE计算结果
        self.advantages = None
        self.returns = None
        
        self.ptr = 0
        self.path_start_idx = 0
    
    def store_step(self, observation, action, log_prob, value):
        """存储一步经验"""
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr += 1
    
    def store_reward_done(self, reward, done):
        """存储奖励和完成标志"""
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done:
            self.path_start_idx = self.ptr
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return len(self.observations) >= self.buffer_size
    
    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float):
        """计算优势和回报"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # 计算GAE
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = 0
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]
            
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        # 计算回报
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """获取批次数据"""
        # 转换观测
        if isinstance(self.observations[0], dict):
            obs_batch = {}
            for key in self.observations[0].keys():
                if isinstance(self.observations[0][key], dict):
                    # 嵌套字典
                    obs_batch[key] = {}
                    for sub_key in self.observations[0][key].keys():
                        obs_batch[key][sub_key] = torch.FloatTensor(
                            [obs[key][sub_key] for obs in self.observations]
                        ).to(self.device)
                else:
                    obs_batch[key] = torch.FloatTensor(
                        [obs[key] for obs in self.observations]
                    ).to(self.device)
        else:
            obs_batch = torch.FloatTensor(self.observations).to(self.device)
        
        batch_data = {
            'observations': obs_batch,
            'actions': torch.FloatTensor(self.actions).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'advantages': torch.FloatTensor(self.advantages).to(self.device),
            'returns': torch.FloatTensor(self.returns).to(self.device)
        }
        
        return batch_data
    
    def clear(self):
        """清空缓冲区"""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.ptr = 0
        self.path_start_idx = 0