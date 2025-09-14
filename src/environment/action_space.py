"""
动作空间处理器 - 处理连续和离散动作空间的转换
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Union
import gymnasium as gym
from gymnasium import spaces


class ActionSpace:
    """动作空间处理器类"""
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        初始化动作空间处理器
        
        Args:
            env_config: 环境配置
        """
        self.action_type = env_config.get('action_type', 'continuous')  # continuous or discrete
        
        if self.action_type == 'continuous':
            self.action_bounds = env_config.get('action_bounds', {
                'velocity_x': [-5.0, 5.0],
                'velocity_y': [-5.0, 5.0],
                'velocity_z': [-2.0, 2.0],
                'yaw_rate': [-90.0, 90.0]
            })
            self.action_dim = 4  # vx, vy, vz, yaw_rate
        else:  # discrete
            self.discrete_actions = env_config.get('discrete_actions', [
                [0, 0, 0, 0],      # hover
                [2, 0, 0, 0],      # forward
                [-2, 0, 0, 0],     # backward
                [0, 2, 0, 0],      # right
                [0, -2, 0, 0],     # left
                [0, 0, 1, 0],      # up
                [0, 0, -1, 0],     # down
                [0, 0, 0, 30],     # turn right
                [0, 0, 0, -30]     # turn left
            ])
            self.action_dim = len(self.discrete_actions)
        
        # 动作平滑参数
        self.enable_action_smoothing = True
        self.smoothing_factor = 0.8  # 当前动作权重
        self.previous_action = np.zeros(4)
        
        # 安全限制
        self.enable_safety_limits = True
        self.max_velocity_change = 2.0  # 最大速度变化率 (m/s per step)
        self.max_yaw_rate = 90.0  # 最大偏航率 (deg/s)
        
        # 创建Gym动作空间
        self.gym_space = self._create_gym_space()
    
    def _create_gym_space(self) -> gym.Space:
        """
        创建Gym兼容的动作空间
        
        Returns:
            Gym动作空间
        """
        if self.action_type == 'continuous':
            # 连续动作空间 [vx, vy, vz, yaw_rate]
            low = np.array([
                self.action_bounds['velocity_x'][0],
                self.action_bounds['velocity_y'][0],
                self.action_bounds['velocity_z'][0],
                self.action_bounds['yaw_rate'][0]
            ], dtype=np.float32)
            
            high = np.array([
                self.action_bounds['velocity_x'][1],
                self.action_bounds['velocity_y'][1],
                self.action_bounds['velocity_z'][1],
                self.action_bounds['yaw_rate'][1]
            ], dtype=np.float32)
            
            return spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            # 离散动作空间
            return spaces.Discrete(self.action_dim)
    
    def get_gym_space(self) -> gym.Space:
        """获取Gym动作空间"""
        return self.gym_space
    
    def process_action(self, action: Union[np.ndarray, int, List[float]]) -> np.ndarray:
        """
        处理算法输出的动作
        
        Args:
            action: 原始动作 (连续动作向量或离散动作索引)
            
        Returns:
            处理后的速度命令 [vx, vy, vz, yaw_rate]
        """
        if self.action_type == 'continuous':
            velocity_command = self._process_continuous_action(action)
        else:
            velocity_command = self._process_discrete_action(action)
        
        # 应用安全限制
        if self.enable_safety_limits:
            velocity_command = self._apply_safety_limits(velocity_command)
        
        # 应用动作平滑
        if self.enable_action_smoothing:
            velocity_command = self._apply_action_smoothing(velocity_command)
        
        # 更新历史动作
        self.previous_action = velocity_command.copy()
        
        return velocity_command
    
    def _process_continuous_action(self, action: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        处理连续动作
        
        Args:
            action: 连续动作向量
            
        Returns:
            速度命令
        """
        if isinstance(action, list):
            action = np.array(action)
        
        # 确保动作维度正确
        if len(action) != 4:
            raise ValueError(f"Expected 4-dimensional action, got {len(action)}")
        
        # 裁剪到动作边界
        velocity_command = np.array([
            np.clip(action[0], self.action_bounds['velocity_x'][0], self.action_bounds['velocity_x'][1]),
            np.clip(action[1], self.action_bounds['velocity_y'][0], self.action_bounds['velocity_y'][1]),
            np.clip(action[2], self.action_bounds['velocity_z'][0], self.action_bounds['velocity_z'][1]),
            np.clip(action[3], self.action_bounds['yaw_rate'][0], self.action_bounds['yaw_rate'][1])
        ], dtype=np.float32)
        
        return velocity_command
    
    def _process_discrete_action(self, action: Union[int, np.ndarray]) -> np.ndarray:
        """
        处理离散动作
        
        Args:
            action: 离散动作索引
            
        Returns:
            速度命令
        """
        # 处理不同输入格式
        if isinstance(action, np.ndarray):
            if action.shape == ():  # 标量数组
                action_idx = int(action.item())
            elif len(action) == 1:  # 单元素数组
                action_idx = int(action[0])
            else:
                raise ValueError(f"Invalid discrete action format: {action}")
        else:
            action_idx = int(action)
        
        # 检查动作索引有效性
        if not (0 <= action_idx < len(self.discrete_actions)):
            raise ValueError(f"Action index {action_idx} out of range [0, {len(self.discrete_actions)})")
        
        # 获取对应的速度命令
        velocity_command = np.array(self.discrete_actions[action_idx], dtype=np.float32)
        
        return velocity_command
    
    def _apply_safety_limits(self, velocity_command: np.ndarray) -> np.ndarray:
        """
        应用安全限制
        
        Args:
            velocity_command: 原始速度命令
            
        Returns:
            安全限制后的速度命令
        """
        limited_command = velocity_command.copy()
        
        # 限制速度变化率（防止过于剧烈的动作变化）
        velocity_change = velocity_command[:3] - self.previous_action[:3]
        velocity_change_magnitude = np.linalg.norm(velocity_change)
        
        if velocity_change_magnitude > self.max_velocity_change:
            # 缩放速度变化
            scale_factor = self.max_velocity_change / velocity_change_magnitude
            limited_change = velocity_change * scale_factor
            limited_command[:3] = self.previous_action[:3] + limited_change
        
        # 限制偏航率
        limited_command[3] = np.clip(limited_command[3], -self.max_yaw_rate, self.max_yaw_rate)
        
        # 确保在全局边界内
        if self.action_type == 'continuous':
            limited_command[0] = np.clip(limited_command[0], 
                                       self.action_bounds['velocity_x'][0], 
                                       self.action_bounds['velocity_x'][1])
            limited_command[1] = np.clip(limited_command[1],
                                       self.action_bounds['velocity_y'][0], 
                                       self.action_bounds['velocity_y'][1])
            limited_command[2] = np.clip(limited_command[2],
                                       self.action_bounds['velocity_z'][0], 
                                       self.action_bounds['velocity_z'][1])
        
        return limited_command
    
    def _apply_action_smoothing(self, velocity_command: np.ndarray) -> np.ndarray:
        """
        应用动作平滑
        
        Args:
            velocity_command: 原始速度命令
            
        Returns:
            平滑后的速度命令
        """
        # 指数移动平均
        smoothed_command = (self.smoothing_factor * velocity_command + 
                           (1.0 - self.smoothing_factor) * self.previous_action)
        
        return smoothed_command.astype(np.float32)
    
    def reset_action_history(self):
        """重置动作历史（用于环境重置）"""
        self.previous_action = np.zeros(4)
    
    def get_action_info(self) -> Dict[str, Any]:
        """
        获取动作空间信息
        
        Returns:
            动作空间信息字典
        """
        info = {
            'action_type': self.action_type,
            'action_dim': self.action_dim,
            'enable_action_smoothing': self.enable_action_smoothing,
            'enable_safety_limits': self.enable_safety_limits
        }
        
        if self.action_type == 'continuous':
            info['action_bounds'] = self.action_bounds
            info['smoothing_factor'] = self.smoothing_factor
            info['max_velocity_change'] = self.max_velocity_change
        else:
            info['discrete_actions'] = self.discrete_actions
            info['action_meanings'] = [
                'hover', 'forward', 'backward', 'right', 'left', 
                'up', 'down', 'turn_right', 'turn_left'
            ]
        
        info['max_yaw_rate'] = self.max_yaw_rate
        
        return info
    
    def sample_random_action(self) -> Union[np.ndarray, int]:
        """
        采样随机动作
        
        Returns:
            随机动作
        """
        if self.action_type == 'continuous':
            action = np.array([
                np.random.uniform(self.action_bounds['velocity_x'][0], 
                                self.action_bounds['velocity_x'][1]),
                np.random.uniform(self.action_bounds['velocity_y'][0], 
                                self.action_bounds['velocity_y'][1]),
                np.random.uniform(self.action_bounds['velocity_z'][0], 
                                self.action_bounds['velocity_z'][1]),
                np.random.uniform(self.action_bounds['yaw_rate'][0], 
                                self.action_bounds['yaw_rate'][1])
            ], dtype=np.float32)
            return action
        else:
            return np.random.randint(0, self.action_dim)
    
    def action_to_string(self, action: Union[np.ndarray, int]) -> str:
        """
        将动作转换为可读字符串
        
        Args:
            action: 动作
            
        Returns:
            动作描述字符串
        """
        if self.action_type == 'continuous':
            if isinstance(action, (int, np.integer)):
                return f"Invalid continuous action: {action}"
            return f"vel_x: {action[0]:.2f}, vel_y: {action[1]:.2f}, vel_z: {action[2]:.2f}, yaw_rate: {action[3]:.2f}"
        else:
            if isinstance(action, np.ndarray):
                if action.shape == ():
                    action_idx = int(action.item())
                else:
                    action_idx = int(action[0])
            else:
                action_idx = int(action)
            
            if 0 <= action_idx < len(self.discrete_actions):
                action_meanings = [
                    'hover', 'forward', 'backward', 'right', 'left', 
                    'up', 'down', 'turn_right', 'turn_left'
                ]
                meaning = action_meanings[action_idx] if action_idx < len(action_meanings) else f'action_{action_idx}'
                command = self.discrete_actions[action_idx]
                return f"{meaning}: [{command[0]}, {command[1]}, {command[2]}, {command[3]}]"
            else:
                return f"Invalid action index: {action_idx}"
    
    def get_action_bounds_normalized(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取归一化的动作边界（用于某些算法）
        
        Returns:
            (low_bounds, high_bounds)
        """
        if self.action_type == 'continuous':
            low = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            return low, high
        else:
            # 离散动作空间不需要归一化边界
            return None, None
    
    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        将归一化动作[-1,1]转换为实际动作边界
        
        Args:
            normalized_action: 归一化动作
            
        Returns:
            实际动作
        """
        if self.action_type != 'continuous':
            raise ValueError("Denormalization only applies to continuous actions")
        
        # 从[-1,1]映射到实际边界
        denormalized = np.array([
            np.interp(normalized_action[0], [-1, 1], self.action_bounds['velocity_x']),
            np.interp(normalized_action[1], [-1, 1], self.action_bounds['velocity_y']),
            np.interp(normalized_action[2], [-1, 1], self.action_bounds['velocity_z']),
            np.interp(normalized_action[3], [-1, 1], self.action_bounds['yaw_rate'])
        ], dtype=np.float32)
        
        return denormalized