"""
奖励组件 - 各种奖励计算的具体实现
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..utils.logger import get_logger


class NavigationRewardComponent:
    """导航奖励组件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化导航奖励组件
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = get_logger("navigation_reward")
        
        # 导航奖励参数
        self.distance_reward_scale = self.config.get('distance_reward_scale', 1.0)
        self.direction_reward_scale = self.config.get('direction_reward_scale', 0.5)
        self.progress_reward_scale = self.config.get('progress_reward_scale', 2.0)
        
        # 距离函数类型
        self.distance_function = self.config.get('distance_function', 'linear')  # linear, exponential, log
        
    def calculate_reward(self, current_position: np.ndarray, 
                        target_position: np.ndarray,
                        previous_position: np.ndarray,
                        step: int) -> Tuple[float, Dict[str, float]]:
        """
        计算导航奖励
        
        Args:
            current_position: 当前位置
            target_position: 目标位置
            previous_position: 上一步位置
            step: 当前步数
            
        Returns:
            (奖励值, 详细信息字典)
        """
        # 距离相关计算
        current_distance = np.linalg.norm(current_position - target_position)
        previous_distance = np.linalg.norm(previous_position - target_position)
        
        # 1. 距离奖励（越近越好）
        distance_reward = self._calculate_distance_reward(current_distance)
        
        # 2. 方向奖励（朝目标方向移动）
        direction_reward = self._calculate_direction_reward(
            current_position, previous_position, target_position
        )
        
        # 3. 进步奖励（距离减少）
        progress_reward = self._calculate_progress_reward(
            current_distance, previous_distance
        )
        
        # 总导航奖励
        total_reward = (distance_reward * self.distance_reward_scale +
                       direction_reward * self.direction_reward_scale +
                       progress_reward * self.progress_reward_scale)
        
        # 详细信息
        info = {
            'distance_reward': distance_reward,
            'direction_reward': direction_reward,  
            'progress_reward': progress_reward,
            'current_distance': current_distance,
            'distance_change': previous_distance - current_distance
        }
        
        return total_reward, info
    
    def _calculate_distance_reward(self, distance: float) -> float:
        """计算基于距离的奖励"""
        if self.distance_function == 'linear':
            # 线性递减奖励，距离越远奖励越小
            max_distance = 100.0  # 假设最大距离
            return max(0.0, 1.0 - distance / max_distance)
        
        elif self.distance_function == 'exponential':
            # 指数递减奖励
            return np.exp(-distance / 20.0)
        
        elif self.distance_function == 'log':
            # 对数奖励
            return -np.log(max(distance, 0.1)) / 10.0
        
        else:
            return -distance / 50.0  # 默认线性惩罚
    
    def _calculate_direction_reward(self, current_pos: np.ndarray, 
                                   previous_pos: np.ndarray,
                                   target_pos: np.ndarray) -> float:
        """计算方向奖励（是否朝目标移动）"""
        # 移动向量
        movement = current_pos - previous_pos
        movement_magnitude = np.linalg.norm(movement)
        
        if movement_magnitude < 1e-6:
            return 0.0  # 没有移动
        
        # 目标方向向量
        to_target = target_pos - previous_pos
        to_target_magnitude = np.linalg.norm(to_target)
        
        if to_target_magnitude < 1e-6:
            return 0.0  # 已经在目标位置
        
        # 计算方向相似度（余弦相似度）
        dot_product = np.dot(movement, to_target)
        cosine_similarity = dot_product / (movement_magnitude * to_target_magnitude)
        
        # 转换为奖励 [0, 1]
        direction_reward = (cosine_similarity + 1.0) / 2.0
        
        return direction_reward
    
    def _calculate_progress_reward(self, current_distance: float, 
                                  previous_distance: float) -> float:
        """计算进步奖励"""
        distance_change = previous_distance - current_distance
        
        # 距离减少给正奖励，增加给负奖励
        if distance_change > 0:
            # 越接近目标，进步奖励越大
            progress_reward = distance_change * (1.0 + 1.0 / max(current_distance, 1.0))
        else:
            # 远离目标的惩罚
            progress_reward = distance_change * 0.5
        
        return progress_reward


class SafetyRewardComponent:
    """安全奖励组件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化安全奖励组件
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = get_logger("safety_reward")
        
        # 安全奖励参数
        self.collision_penalty = self.config.get('collision_penalty', -10.0)
        self.boundary_penalty = self.config.get('boundary_penalty', -5.0)
        self.height_penalty_scale = self.config.get('height_penalty_scale', 1.0)
        self.obstacle_distance_reward_scale = self.config.get('obstacle_distance_reward_scale', 0.1)
        
        # 安全边界
        self.min_height = self.config.get('min_height', 2.0)
        self.max_height = self.config.get('max_height', 15.0)
        self.safe_distance_threshold = self.config.get('safe_distance_threshold', 2.0)
        
    def calculate_reward(self, current_position: np.ndarray,
                        is_collision: bool,
                        is_out_of_bounds: bool,
                        step: int) -> Tuple[float, Dict[str, float]]:
        """
        计算安全奖励
        
        Args:
            current_position: 当前位置
            is_collision: 是否碰撞
            is_out_of_bounds: 是否越界
            step: 当前步数
            
        Returns:
            (奖励值, 详细信息字典)
        """
        total_reward = 0.0
        info = {}
        
        # 1. 碰撞惩罚
        if is_collision:
            collision_reward = self.collision_penalty
            total_reward += collision_reward
            info['collision_penalty'] = collision_reward
            self.logger.debug(f"Collision detected, penalty: {collision_reward}")
        
        # 2. 边界惩罚  
        if is_out_of_bounds:
            boundary_reward = self.boundary_penalty
            total_reward += boundary_reward
            info['boundary_penalty'] = boundary_reward
            self.logger.debug(f"Out of bounds, penalty: {boundary_reward}")
        
        # 3. 高度安全奖励
        height_reward = self._calculate_height_reward(current_position[2])
        total_reward += height_reward
        info['height_reward'] = height_reward
        
        # 4. 基础安全奖励（没有碰撞时给小奖励）
        if not is_collision and not is_out_of_bounds:
            safety_bonus = 0.1
            total_reward += safety_bonus
            info['safety_bonus'] = safety_bonus
        
        return total_reward, info
    
    def _calculate_height_reward(self, z_position: float) -> float:
        """
        计算高度相关的安全奖励
        
        Args:
            z_position: Z位置（AirSim中负值表示高度）
            
        Returns:
            高度奖励
        """
        height = -z_position  # 转换为正高度值
        
        # 在安全高度范围内给奖励
        if self.min_height <= height <= self.max_height:
            # 在中间高度给最高奖励
            mid_height = (self.min_height + self.max_height) / 2
            distance_from_mid = abs(height - mid_height)
            max_distance = (self.max_height - self.min_height) / 2
            
            # 越接近中间高度奖励越高
            height_reward = (1.0 - distance_from_mid / max_distance) * 0.1
        else:
            # 超出安全范围的惩罚
            if height < self.min_height:
                # 过低惩罚
                height_reward = -(self.min_height - height) * self.height_penalty_scale
            else:
                # 过高惩罚
                height_reward = -(height - self.max_height) * self.height_penalty_scale
        
        return height_reward


class EfficiencyRewardComponent:
    """效率奖励组件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化效率奖励组件
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = get_logger("efficiency_reward")
        
        # 效率奖励参数
        self.step_penalty = self.config.get('step_penalty', -0.01)
        self.velocity_efficiency_scale = self.config.get('velocity_efficiency_scale', 0.1)
        self.path_efficiency_scale = self.config.get('path_efficiency_scale', 0.1)
        
        # 效率计算历史
        self.position_history = []
        self.max_history_length = 50
        
    def calculate_reward(self, current_position: np.ndarray,
                        target_position: np.ndarray,
                        action: np.ndarray,
                        step: int) -> Tuple[float, Dict[str, float]]:
        """
        计算效率奖励
        
        Args:
            current_position: 当前位置
            target_position: 目标位置
            action: 执行的动作
            step: 当前步数
            
        Returns:
            (奖励值, 详细信息字典)
        """
        # 更新位置历史
        self.position_history.append(current_position.copy())
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        total_reward = 0.0
        info = {}
        
        # 1. 步数惩罚（鼓励快速完成任务）
        step_reward = self.step_penalty
        total_reward += step_reward
        info['step_penalty'] = step_reward
        
        # 2. 速度效率奖励
        velocity_reward = self._calculate_velocity_efficiency(action, current_position, target_position)
        total_reward += velocity_reward * self.velocity_efficiency_scale
        info['velocity_efficiency'] = velocity_reward
        
        # 3. 路径效率奖励
        if len(self.position_history) >= 2:
            path_reward = self._calculate_path_efficiency()
            total_reward += path_reward * self.path_efficiency_scale
            info['path_efficiency'] = path_reward
        
        return total_reward, info
    
    def _calculate_velocity_efficiency(self, action: np.ndarray, 
                                     current_position: np.ndarray,
                                     target_position: np.ndarray) -> float:
        """计算速度效率奖励"""
        # 提取速度分量
        if len(action) >= 3:
            velocity = action[:3]  # vx, vy, vz
            velocity_magnitude = np.linalg.norm(velocity)
            
            if velocity_magnitude < 1e-6:
                return -0.5  # 惩罚过慢的移动
            
            # 目标方向
            to_target = target_position - current_position
            to_target_magnitude = np.linalg.norm(to_target)
            
            if to_target_magnitude < 1e-6:
                return 0.0  # 已在目标位置
            
            # 速度与目标方向的一致性
            to_target_normalized = to_target / to_target_magnitude
            velocity_normalized = velocity / velocity_magnitude
            
            # 方向一致性奖励
            direction_alignment = np.dot(velocity_normalized, to_target_normalized)
            
            # 速度大小奖励（适中速度最好）
            optimal_speed = 3.0  # 最优速度
            speed_efficiency = 1.0 - abs(velocity_magnitude - optimal_speed) / optimal_speed
            speed_efficiency = max(0.0, speed_efficiency)
            
            # 组合奖励
            velocity_reward = (direction_alignment + speed_efficiency) / 2.0
            
            return velocity_reward
        
        return 0.0
    
    def _calculate_path_efficiency(self) -> float:
        """计算路径效率奖励"""
        if len(self.position_history) < 3:
            return 0.0
        
        # 计算最近几步的路径平滑度
        recent_positions = self.position_history[-3:]
        
        # 计算路径弯曲度（角度变化）
        vec1 = recent_positions[1] - recent_positions[0]
        vec2 = recent_positions[2] - recent_positions[1]
        
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        
        if vec1_norm < 1e-6 or vec2_norm < 1e-6:
            return 0.0
        
        # 计算角度变化
        dot_product = np.dot(vec1, vec2)
        cosine_angle = dot_product / (vec1_norm * vec2_norm)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # 角度越小（路径越直）奖励越高
        angle = np.arccos(cosine_angle)
        straightness_reward = 1.0 - angle / np.pi
        
        return straightness_reward
    
    def reset_history(self):
        """重置历史记录（用于新回合）"""
        self.position_history = []


class SmoothnessRewardComponent:
    """平滑性奖励组件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化平滑性奖励组件
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = get_logger("smoothness_reward")
        
        # 平滑性参数
        self.action_smoothness_scale = self.config.get('action_smoothness_scale', 0.1)
        self.velocity_smoothness_scale = self.config.get('velocity_smoothness_scale', 0.1)
        self.max_action_change_penalty = self.config.get('max_action_change_penalty', -1.0)
        
        # 历史记录
        self.previous_action = None
        self.previous_velocity = None
        
    def calculate_reward(self, current_action: np.ndarray,
                        current_velocity: np.ndarray,
                        step: int) -> Tuple[float, Dict[str, float]]:
        """
        计算平滑性奖励
        
        Args:
            current_action: 当前动作
            current_velocity: 当前速度
            step: 当前步数
            
        Returns:
            (奖励值, 详细信息字典)
        """
        total_reward = 0.0
        info = {}
        
        # 1. 动作平滑性奖励
        if self.previous_action is not None:
            action_smoothness = self._calculate_action_smoothness(current_action)
            total_reward += action_smoothness * self.action_smoothness_scale
            info['action_smoothness'] = action_smoothness
        
        # 2. 速度平滑性奖励
        if self.previous_velocity is not None:
            velocity_smoothness = self._calculate_velocity_smoothness(current_velocity)
            total_reward += velocity_smoothness * self.velocity_smoothness_scale
            info['velocity_smoothness'] = velocity_smoothness
        
        # 更新历史
        self.previous_action = current_action.copy()
        self.previous_velocity = current_velocity.copy()
        
        return total_reward, info
    
    def _calculate_action_smoothness(self, current_action: np.ndarray) -> float:
        """计算动作平滑性奖励"""
        action_change = np.linalg.norm(current_action - self.previous_action)
        
        # 动作变化越小，平滑性奖励越高
        max_change = 2.0  # 假设最大合理变化
        smoothness = max(0.0, 1.0 - action_change / max_change)
        
        # 过大的动作变化给额外惩罚
        if action_change > max_change:
            smoothness += self.max_action_change_penalty
        
        return smoothness
    
    def _calculate_velocity_smoothness(self, current_velocity: np.ndarray) -> float:
        """计算速度平滑性奖励"""
        velocity_change = np.linalg.norm(current_velocity - self.previous_velocity)
        
        # 速度变化越小，平滑性奖励越高
        max_velocity_change = 1.0  # 假设最大合理速度变化
        smoothness = max(0.0, 1.0 - velocity_change / max_velocity_change)
        
        return smoothness
    
    def reset_history(self):
        """重置历史记录（用于新回合）"""
        self.previous_action = None
        self.previous_velocity = None