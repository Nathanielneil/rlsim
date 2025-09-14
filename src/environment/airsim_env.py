"""
AirSim环境封装 - 适配Windows-AirSim-UE4.27.2平台的无人机导航环境
"""

import numpy as np
import cv2
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

try:
    import airsim
except ImportError:
    raise ImportError("AirSim Python API not installed. Please install airsim package.")

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger
from .observation_space import ObservationSpace
from .action_space import ActionSpace
from .scene_manager import SceneManager


class AirSimEnv(gym.Env):
    """AirSim无人机导航环境"""
    
    metadata = {'render.modes': ['rgb_array']}
    
    def __init__(self, config_path: Optional[str] = None, algorithm: str = "ppo", 
                 config: Optional[Dict[str, Any]] = None, max_episode_steps: Optional[int] = None, 
                 debug: bool = False):
        """
        初始化AirSim环境
        
        Args:
            config_path: 配置文件路径
            algorithm: 算法名称，用于加载对应配置
            config: 环境配置字典（可选）
            max_episode_steps: 最大回合步数（可选，覆盖配置文件）
            debug: 调试模式（可选）
        """
        super().__init__()
        
        self.debug = debug
        self.logger = get_logger("airsim_env")
        
        # 处理配置
        if config_path or algorithm != "ppo":
            # 使用配置文件加载
            self.config_loader = ConfigLoader(config_path)
            self.airsim_config = self.config_loader.load_airsim_settings()
            self.scene_config = self.config_loader.load_scene_config()
            self.algorithm_config = self.config_loader.load_algorithm_config(algorithm)
        else:
            # 使用默认配置加载器
            self.config_loader = ConfigLoader()
            self.airsim_config = self.config_loader.load_airsim_settings()
            self.scene_config = self.config_loader.load_scene_config()
            self.algorithm_config = self.config_loader.load_algorithm_config(algorithm)
        
        # 环境配置（允许覆盖）
        self.max_episode_steps = max_episode_steps or self.scene_config['environment']['max_episode_steps']
        self.collision_threshold = self.scene_config['environment']['collision_threshold']
        self.success_threshold = self.scene_config['environment']['success_threshold']
        
        # AirSim连接
        self.client = None
        self.vehicle_name = "Drone1"
        self.connect_to_airsim()
        
        # 初始化组件
        self.observation_space_handler = ObservationSpace(self.algorithm_config['env_config'])
        self.action_space_handler = ActionSpace(self.algorithm_config['env_config'])
        self.scene_manager = SceneManager(self.scene_config)
        
        # 设置Gym空间
        self.observation_space = self.observation_space_handler.get_gym_space()
        self.action_space = self.action_space_handler.get_gym_space()
        
        # 环境状态
        self.current_step = 0
        self.episode_reward = 0.0
        self.start_position = np.array([0.0, 0.0, -2.0])
        self.target_position = np.array([0.0, 0.0, -2.0])
        self.previous_position = np.array([0.0, 0.0, -2.0])
        self.is_episode_done = False
        
        # 性能监控
        self.collision_count = 0
        self.trajectory_points = []
        self.action_history = []
        self.reward_history = []
        
        self.logger.info("AirSim environment initialized successfully")
    
    def connect_to_airsim(self, max_retries: int = 5):
        """
        连接到AirSim
        
        Args:
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                
                # 启用API控制
                self.client.enableApiControl(True, self.vehicle_name)
                self.client.armDisarm(True, self.vehicle_name)
                
                self.logger.info(f"Successfully connected to AirSim on attempt {attempt + 1}")
                return
                
            except Exception as e:
                self.logger.warning(f"AirSim connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    self.logger.error("Failed to connect to AirSim after all attempts")
                    raise ConnectionError("Cannot connect to AirSim. Please ensure AirSim is running.")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            初始观测和信息字典
        """
        super().reset(seed=seed)
        
        try:
            # 重置AirSim
            self.client.reset()
            time.sleep(0.5)
            
            # 重新启用控制
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            
            # 设置起始位置
            spawn_config = self.scene_config.get('spawn_positions', [{'position': [0, 0, -2], 'yaw': 0}])
            spawn_idx = np.random.randint(len(spawn_config))
            spawn_info = spawn_config[spawn_idx]
            
            self.start_position = np.array(spawn_info['position'], dtype=np.float32)
            start_yaw = spawn_info['yaw']
            
            # 移动到起始位置
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(self.start_position[0], self.start_position[1], self.start_position[2]),
                    airsim.to_quaternion(0, 0, np.radians(start_yaw))
                ),
                True,
                self.vehicle_name
            )
            
            time.sleep(0.5)
            
            # 生成目标位置
            self.target_position = self.scene_manager.generate_target_position(self.start_position)
            self.previous_position = self.start_position.copy()
            
            # 重置环境状态
            self.current_step = 0
            self.episode_reward = 0.0
            self.is_episode_done = False
            self.collision_count = 0
            self.trajectory_points = [self.start_position.copy()]
            self.action_history = []
            self.reward_history = []
            
            # 获取初始观测
            observation = self._get_observation()
            info = self._get_info()
            
            self.logger.debug(f"Environment reset - Start: {self.start_position}, Target: {self.target_position}")
            
            return observation, info
            
        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            # 尝试重新连接
            self.connect_to_airsim()
            raise
    
    def step(self, action: Union[np.ndarray, List[float]]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作
            
        Returns:
            观测、奖励、是否完成、是否截断、信息字典
        """
        if self.is_episode_done:
            self.logger.warning("Step called on done environment")
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        try:
            # 执行动作
            self._execute_action(action)
            self.current_step += 1
            self.action_history.append(np.array(action))
            
            # 等待动作执行
            time.sleep(0.1)
            
            # 获取新的观测
            observation = self._get_observation()
            current_position = self._get_current_position()
            self.trajectory_points.append(current_position.copy())
            
            # 计算奖励
            reward = self._calculate_reward(current_position, action)
            self.episode_reward += reward
            self.reward_history.append(reward)
            
            # 检查终止条件
            done, truncated, info = self._check_termination(current_position)
            self.is_episode_done = done or truncated
            
            # 更新位置
            self.previous_position = current_position.copy()
            
            return observation, reward, done, truncated, info
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            # 发生错误时终止回合
            return self._get_observation(), -10.0, True, False, {"error": str(e)}
    
    def _execute_action(self, action: Union[np.ndarray, List[float]]):
        """
        执行动作
        
        Args:
            action: 动作向量
        """
        if isinstance(action, list):
            action = np.array(action)
        
        # 通过动作空间处理器处理动作
        velocity_command = self.action_space_handler.process_action(action)
        
        # 执行速度控制
        self.client.moveByVelocityZAsync(
            vx=float(velocity_command[0]),
            vy=float(velocity_command[1]), 
            z=float(self._get_current_position()[2] + velocity_command[2]),
            duration=0.1,
            vehicle_name=self.vehicle_name
        )
        
        # 如果有偏航控制
        if len(velocity_command) > 3 and abs(velocity_command[3]) > 0.1:
            self.client.rotateByYawRateAsync(
                yaw_rate=float(velocity_command[3]),
                duration=0.1,
                vehicle_name=self.vehicle_name
            )
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        获取当前观测
        
        Returns:
            观测字典
        """
        try:
            # 获取图像
            images = self._get_images()
            
            # 获取状态信息
            state = self._get_state()
            
            # 通过观测空间处理器处理
            observation = self.observation_space_handler.process_observation(images, state, self.target_position)
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Failed to get observation: {e}")
            # 返回空观测
            return self.observation_space_handler.get_empty_observation()
    
    def _get_images(self) -> Dict[str, np.ndarray]:
        """
        获取相机图像
        
        Returns:
            图像字典
        """
        try:
            # 请求多种类型的图像
            requests = [
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),           # RGB
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False),       # 深度可视化 
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)    # 语义分割
            ]
            
            responses = self.client.simGetImages(requests, self.vehicle_name)
            
            images = {}
            
            # 处理RGB图像
            if len(responses) > 0 and len(responses[0].image_data_uint8) > 0:
                rgb_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                rgb_img = rgb_img.reshape(responses[0].height, responses[0].width, 3)
                images['rgb'] = cv2.resize(rgb_img, (224, 224))
            else:
                images['rgb'] = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # 处理深度图像
            if len(responses) > 1 and len(responses[1].image_data_uint8) > 0:
                depth_img = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                depth_img = depth_img.reshape(responses[1].height, responses[1].width, 3)
                images['depth'] = cv2.resize(depth_img, (224, 224))
            else:
                images['depth'] = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # 处理分割图像
            if len(responses) > 2 and len(responses[2].image_data_uint8) > 0:
                seg_img = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
                seg_img = seg_img.reshape(responses[2].height, responses[2].width, 3)
                images['segmentation'] = cv2.resize(seg_img, (224, 224))
            else:
                images['segmentation'] = np.zeros((224, 224, 3), dtype=np.uint8)
            
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to get images: {e}")
            return {
                'rgb': np.zeros((224, 224, 3), dtype=np.uint8),
                'depth': np.zeros((224, 224, 3), dtype=np.uint8),
                'segmentation': np.zeros((224, 224, 3), dtype=np.uint8)
            }
    
    def _get_state(self) -> np.ndarray:
        """
        获取无人机状态信息
        
        Returns:
            状态向量 [x, y, z, vx, vy, vz, roll, pitch, yaw, ax, ay, az, target_distance]
        """
        try:
            # 获取多旋翼状态
            state = self.client.getMultirotorState(self.vehicle_name)
            
            # 位置信息
            position = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ])
            
            # 速度信息
            velocity = np.array([
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val
            ])
            
            # 姿态信息（转换为欧拉角）
            orientation = state.kinematics_estimated.orientation
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)
            attitude = np.array([roll, pitch, yaw])
            
            # 加速度信息
            acceleration = np.array([
                state.kinematics_estimated.linear_acceleration.x_val,
                state.kinematics_estimated.linear_acceleration.y_val,
                state.kinematics_estimated.linear_acceleration.z_val
            ])
            
            # 到目标的距离
            target_distance = np.linalg.norm(position - self.target_position)
            
            # 组合状态向量 (13维)
            state_vector = np.concatenate([
                position,        # 3
                velocity,        # 3  
                attitude,        # 3
                acceleration,    # 3
                [target_distance] # 1
            ]).astype(np.float32)
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"Failed to get state: {e}")
            return np.zeros(13, dtype=np.float32)
    
    def _get_current_position(self) -> np.ndarray:
        """
        获取当前位置
        
        Returns:
            位置向量 [x, y, z]
        """
        try:
            state = self.client.getMultirotorState(self.vehicle_name)
            position = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ], dtype=np.float32)
            return position
        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            return self.previous_position.copy()
    
    def _calculate_reward(self, current_position: np.ndarray, action: np.ndarray) -> float:
        """
        计算奖励
        
        Args:
            current_position: 当前位置
            action: 执行的动作
            
        Returns:
            奖励值
        """
        from ..reward.reward_function import RewardFunction
        
        if not hasattr(self, 'reward_function'):
            reward_weights = self.algorithm_config.get('reward_weights', {})
            self.reward_function = RewardFunction(reward_weights)
        
        # 检查碰撞
        collision_info = self.client.simGetCollisionInfo(self.vehicle_name)
        is_collision = collision_info.has_collided
        
        if is_collision:
            self.collision_count += 1
        
        # 计算奖励
        reward = self.reward_function.calculate_reward(
            current_position=current_position,
            target_position=self.target_position,
            previous_position=self.previous_position,
            action=action,
            is_collision=is_collision,
            step=self.current_step
        )
        
        return reward
    
    def _check_termination(self, current_position: np.ndarray) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        检查终止条件
        
        Args:
            current_position: 当前位置
            
        Returns:
            (done, truncated, info)
        """
        info = {}
        done = False
        truncated = False
        
        # 检查成功条件
        distance_to_target = np.linalg.norm(current_position - self.target_position)
        if distance_to_target <= self.success_threshold:
            done = True
            info['success'] = True
            info['termination_reason'] = 'target_reached'
        
        # 检查碰撞
        collision_info = self.client.simGetCollisionInfo(self.vehicle_name)
        if collision_info.has_collided:
            done = True
            info['success'] = False
            info['termination_reason'] = 'collision'
        
        # 检查边界
        bounds = self.scene_config['scene_bounds']
        if (current_position[0] < bounds['x_min'] or current_position[0] > bounds['x_max'] or
            current_position[1] < bounds['y_min'] or current_position[1] > bounds['y_max'] or
            current_position[2] < bounds['z_min'] or current_position[2] > bounds['z_max']):
            done = True
            info['success'] = False
            info['termination_reason'] = 'out_of_bounds'
        
        # 检查最大步数
        if self.current_step >= self.max_episode_steps:
            truncated = True
            info['success'] = False
            info['termination_reason'] = 'max_steps'
        
        return done, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """
        获取信息字典
        
        Returns:
            信息字典
        """
        current_position = self._get_current_position()
        distance_to_target = np.linalg.norm(current_position - self.target_position)
        
        info = {
            'current_position': current_position,
            'target_position': self.target_position,
            'distance_to_target': distance_to_target,
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'collision_count': self.collision_count,
            'trajectory_length': len(self.trajectory_points)
        }
        
        return info
    
    def render(self, mode='rgb_array'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
            
        Returns:
            渲染图像
        """
        if mode == 'rgb_array':
            images = self._get_images()
            return images.get('rgb', np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")
    
    def close(self):
        """关闭环境"""
        try:
            if self.client:
                self.client.enableApiControl(False, self.vehicle_name)
                self.client.armDisarm(False, self.vehicle_name)
                self.logger.info("AirSim environment closed")
        except Exception as e:
            self.logger.error(f"Error closing environment: {e}")
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        获取回合统计信息
        
        Returns:
            统计信息字典
        """
        if len(self.trajectory_points) < 2:
            return {}
        
        trajectory = np.array(self.trajectory_points)
        
        # 轨迹长度
        trajectory_length = 0.0
        for i in range(1, len(trajectory)):
            trajectory_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        # 最终距离误差
        final_distance = np.linalg.norm(trajectory[-1] - self.target_position)
        
        # Oracle成功率（是否曾经接近过目标）
        min_distance = float('inf')
        for point in trajectory:
            distance = np.linalg.norm(point - self.target_position)
            min_distance = min(min_distance, distance)
        
        oracle_success = min_distance <= self.success_threshold
        
        # 成功率
        success = final_distance <= self.success_threshold
        
        # 计算SPL（路径长度优化的成功率）
        optimal_length = np.linalg.norm(self.target_position - self.start_position)
        spl = 0.0
        if success and trajectory_length > 0:
            spl = optimal_length / max(optimal_length, trajectory_length)
        
        statistics = {
            'trajectory_length': trajectory_length,
            'navigation_error': final_distance,
            'oracle_success_rate': oracle_success,
            'success_rate': success,
            'spl': spl,
            'collision_count': self.collision_count,
            'episode_steps': self.current_step,
            'total_reward': self.episode_reward
        }
        
        return statistics


# 为向后兼容性添加别名
AirSimNavigationEnv = AirSimEnv