"""
观测空间处理器 - 处理视觉观测和状态信息的组合
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Union
import gymnasium as gym
from gymnasium import spaces


class ObservationSpace:
    """观测空间处理器类"""
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        初始化观测空间处理器
        
        Args:
            env_config: 环境配置
        """
        self.observation_type = env_config.get('observation_type', 'mixed')  # vision, state, mixed
        self.image_size = (224, 224)  # AirSim配置的图像尺寸
        self.state_dim = 13  # 状态向量维度
        
        # 图像预处理参数
        self.normalize_images = True
        self.image_mean = np.array([0.485, 0.456, 0.406])  # ImageNet均值
        self.image_std = np.array([0.229, 0.224, 0.225])   # ImageNet标准差
        
        # 状态归一化参数
        self.state_bounds = {
            'position': [-100.0, 100.0],      # x, y, z位置范围
            'velocity': [-10.0, 10.0],        # 速度范围
            'attitude': [-np.pi, np.pi],      # 姿态角范围
            'acceleration': [-20.0, 20.0],    # 加速度范围
            'distance': [0.0, 150.0]          # 距离范围
        }
        
        # 创建Gym观测空间
        self.gym_space = self._create_gym_space()
    
    def _create_gym_space(self) -> gym.Space:
        """
        创建Gym兼容的观测空间
        
        Returns:
            Gym观测空间
        """
        if self.observation_type == 'vision':
            # 仅视觉观测：RGB + 深度 + 分割图像
            return spaces.Dict({
                'rgb': spaces.Box(
                    low=0, high=255, 
                    shape=(self.image_size[1], self.image_size[0], 3), 
                    dtype=np.uint8
                ),
                'depth': spaces.Box(
                    low=0, high=255,
                    shape=(self.image_size[1], self.image_size[0], 3),
                    dtype=np.uint8
                ),
                'segmentation': spaces.Box(
                    low=0, high=255,
                    shape=(self.image_size[1], self.image_size[0], 3),
                    dtype=np.uint8
                )
            })
            
        elif self.observation_type == 'state':
            # 仅状态观测
            return spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim + 3,),  # 状态 + 目标位置
                dtype=np.float32
            )
            
        else:  # mixed
            # 混合观测：视觉 + 状态
            return spaces.Dict({
                'images': spaces.Dict({
                    'rgb': spaces.Box(
                        low=0.0, high=1.0,
                        shape=(3, self.image_size[1], self.image_size[0]),
                        dtype=np.float32
                    ),
                    'depth': spaces.Box(
                        low=0.0, high=1.0,
                        shape=(1, self.image_size[1], self.image_size[0]),
                        dtype=np.float32
                    ),
                    'segmentation': spaces.Box(
                        low=0.0, high=1.0,
                        shape=(3, self.image_size[1], self.image_size[0]),
                        dtype=np.float32
                    )
                }),
                'state': spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self.state_dim + 6,),  # 状态 + 目标位置 + 相对位置
                    dtype=np.float32
                )
            })
    
    def get_gym_space(self) -> gym.Space:
        """获取Gym观测空间"""
        return self.gym_space
    
    def process_observation(self, images: Dict[str, np.ndarray], 
                          state: np.ndarray, 
                          target_position: np.ndarray) -> Dict[str, Any]:
        """
        处理原始观测数据
        
        Args:
            images: 图像字典 {'rgb': ndarray, 'depth': ndarray, 'segmentation': ndarray}
            state: 状态向量 [x,y,z,vx,vy,vz,roll,pitch,yaw,ax,ay,az,target_dist]
            target_position: 目标位置 [x,y,z]
            
        Returns:
            处理后的观测字典
        """
        if self.observation_type == 'vision':
            return self._process_vision_only(images)
        elif self.observation_type == 'state':
            return self._process_state_only(state, target_position)
        else:  # mixed
            return self._process_mixed_observation(images, state, target_position)
    
    def _process_vision_only(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        处理仅视觉观测
        
        Args:
            images: 原始图像字典
            
        Returns:
            处理后的视觉观测
        """
        processed_images = {}
        
        for image_type, image in images.items():
            if image is None or image.size == 0:
                # 创建空图像
                processed_images[image_type] = np.zeros(
                    (self.image_size[1], self.image_size[0], 3), 
                    dtype=np.uint8
                )
            else:
                # 确保图像尺寸正确
                if image.shape[:2] != self.image_size:
                    image = cv2.resize(image, self.image_size)
                
                # 确保是3通道
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
                processed_images[image_type] = image.astype(np.uint8)
        
        return processed_images
    
    def _process_state_only(self, state: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        """
        处理仅状态观测
        
        Args:
            state: 原始状态向量
            target_position: 目标位置
            
        Returns:
            处理后的状态观测
        """
        # 组合状态和目标位置
        combined_state = np.concatenate([state, target_position])
        
        # 归一化处理
        normalized_state = self._normalize_state(combined_state)
        
        return normalized_state.astype(np.float32)
    
    def _process_mixed_observation(self, images: Dict[str, np.ndarray], 
                                 state: np.ndarray, 
                                 target_position: np.ndarray) -> Dict[str, Any]:
        """
        处理混合观测
        
        Args:
            images: 原始图像字典
            state: 原始状态向量
            target_position: 目标位置
            
        Returns:
            处理后的混合观测
        """
        # 处理图像
        processed_images = {}
        
        for image_type, image in images.items():
            if image is None or image.size == 0:
                # 创建空图像
                if image_type == 'depth':
                    processed_images[image_type] = np.zeros(
                        (1, self.image_size[1], self.image_size[0]), 
                        dtype=np.float32
                    )
                else:
                    processed_images[image_type] = np.zeros(
                        (3, self.image_size[1], self.image_size[0]), 
                        dtype=np.float32
                    )
            else:
                processed_images[image_type] = self._preprocess_image(image, image_type)
        
        # 处理状态
        current_position = state[:3]  # 当前位置
        relative_position = target_position - current_position  # 相对位置
        
        # 组合状态：原始状态 + 目标位置 + 相对位置
        combined_state = np.concatenate([state, target_position, relative_position])
        normalized_state = self._normalize_state(combined_state)
        
        return {
            'images': processed_images,
            'state': normalized_state.astype(np.float32)
        }
    
    def _preprocess_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """
        预处理单张图像
        
        Args:
            image: 原始图像
            image_type: 图像类型
            
        Returns:
            预处理后的图像 (C, H, W)
        """
        # 确保图像尺寸正确
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        
        # 转换为float并归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        
        # 不同类型图像的特殊处理
        if image_type == 'rgb':
            # RGB图像标准化
            if self.normalize_images:
                image = (image - self.image_mean) / self.image_std
                # 裁剪到合理范围
                image = np.clip(image, -2.0, 2.0)
                # 重新归一化到[0,1]
                image = (image + 2.0) / 4.0
            
            # 转换为CHW格式
            image = np.transpose(image, (2, 0, 1))
            
        elif image_type == 'depth':
            # 深度图像处理
            if len(image.shape) == 3:
                # 如果是3通道深度图，转换为单通道
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 添加通道维度并转换为CHW
            image = np.expand_dims(image, axis=0)
            
        elif image_type == 'segmentation':
            # 分割图像保持3通道
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # 转换为CHW格式
            image = np.transpose(image, (2, 0, 1))
        
        return image.astype(np.float32)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        归一化状态向量
        
        Args:
            state: 原始状态向量
            
        Returns:
            归一化后的状态向量
        """
        normalized = state.copy()
        
        # 位置 (前3个)
        normalized[:3] = self._normalize_values(
            state[:3], 
            self.state_bounds['position'][0], 
            self.state_bounds['position'][1]
        )
        
        # 速度 (4-6)
        if len(state) > 6:
            normalized[3:6] = self._normalize_values(
                state[3:6],
                self.state_bounds['velocity'][0],
                self.state_bounds['velocity'][1]
            )
        
        # 姿态角 (7-9)
        if len(state) > 9:
            normalized[6:9] = self._normalize_values(
                state[6:9],
                self.state_bounds['attitude'][0],
                self.state_bounds['attitude'][1]
            )
        
        # 加速度 (10-12)
        if len(state) > 12:
            normalized[9:12] = self._normalize_values(
                state[9:12],
                self.state_bounds['acceleration'][0],
                self.state_bounds['acceleration'][1]
            )
        
        # 距离信息
        if len(state) > 13:
            # 目标距离
            normalized[12] = self._normalize_values(
                state[12:13],
                self.state_bounds['distance'][0],
                self.state_bounds['distance'][1]
            )[0]
            
            # 目标位置 (14-16)
            if len(state) > 16:
                normalized[13:16] = self._normalize_values(
                    state[13:16],
                    self.state_bounds['position'][0],
                    self.state_bounds['position'][1]
                )
            
            # 相对位置 (17-19)
            if len(state) > 19:
                normalized[16:19] = self._normalize_values(
                    state[16:19],
                    -2 * max(abs(self.state_bounds['position'][0]), abs(self.state_bounds['position'][1])),
                    2 * max(abs(self.state_bounds['position'][0]), abs(self.state_bounds['position'][1]))
                )
        
        return normalized
    
    def _normalize_values(self, values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """
        将数值归一化到[-1, 1]范围
        
        Args:
            values: 原始数值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            归一化后的数值
        """
        # 避免除零
        if max_val == min_val:
            return np.zeros_like(values)
        
        # 归一化到[-1, 1]
        normalized = 2.0 * (values - min_val) / (max_val - min_val) - 1.0
        
        # 裁剪到合理范围
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def get_empty_observation(self) -> Dict[str, Any]:
        """
        获取空观测（用于错误处理）
        
        Returns:
            空观测字典
        """
        if self.observation_type == 'vision':
            return {
                'rgb': np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
                'depth': np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
                'segmentation': np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            }
        elif self.observation_type == 'state':
            return np.zeros(self.state_dim + 3, dtype=np.float32)
        else:  # mixed
            return {
                'images': {
                    'rgb': np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32),
                    'depth': np.zeros((1, self.image_size[1], self.image_size[0]), dtype=np.float32),
                    'segmentation': np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32)
                },
                'state': np.zeros(self.state_dim + 6, dtype=np.float32)
            }
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        获取观测空间信息
        
        Returns:
            观测空间信息字典
        """
        info = {
            'observation_type': self.observation_type,
            'image_size': self.image_size,
            'state_dim': self.state_dim,
            'normalize_images': self.normalize_images
        }
        
        if self.observation_type in ['mixed', 'vision']:
            info['image_channels'] = {
                'rgb': 3,
                'depth': 1 if self.observation_type == 'mixed' else 3,
                'segmentation': 3
            }
        
        if self.observation_type in ['mixed', 'state']:
            info['state_components'] = {
                'position': [0, 3],
                'velocity': [3, 6],  
                'attitude': [6, 9],
                'acceleration': [9, 12],
                'target_distance': 12,
                'target_position': [13, 16],
                'relative_position': [16, 19]
            }
        
        return info