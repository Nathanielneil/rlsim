"""
奖励函数 - 统一的多维度奖励计算系统
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .reward_components import (
    NavigationRewardComponent,
    SafetyRewardComponent, 
    EfficiencyRewardComponent,
    SmoothnessRewardComponent
)
from ..utils.logger import get_logger


class RewardFunction:
    """统一奖励函数类"""
    
    def __init__(self, reward_config: Dict[str, Any]):
        """
        初始化奖励函数
        
        Args:
            reward_config: 奖励配置字典
        """
        self.logger = get_logger("reward_function")
        self.reward_config = reward_config
        
        # 奖励权重
        self.weights = {
            'navigation': reward_config.get('navigation', 1.0),
            'safety': reward_config.get('safety', 1.0),
            'efficiency': reward_config.get('efficiency', 0.5),
            'smoothness': reward_config.get('smoothness', 0.3)
        }
        
        # 特殊奖励
        self.collision_penalty = reward_config.get('collision_penalty', -10.0)
        self.success_reward = reward_config.get('success_reward', 100.0)
        self.step_penalty = reward_config.get('step_penalty', -0.01)
        
        # 初始化奖励组件
        self._initialize_components()
        
        # 奖励统计
        self.reward_history = []
        self.component_history = {
            'navigation': [],
            'safety': [],
            'efficiency': [], 
            'smoothness': [],
            'special': []
        }
        
        self.logger.info(f"Reward function initialized with weights: {self.weights}")
    
    def _initialize_components(self):
        """初始化奖励组件"""
        # 导航奖励组件
        nav_config = self.reward_config.get('navigation_config', {})
        self.navigation_component = NavigationRewardComponent(nav_config)
        
        # 安全奖励组件
        safety_config = self.reward_config.get('safety_config', {})
        self.safety_component = SafetyRewardComponent(safety_config)
        
        # 效率奖励组件
        efficiency_config = self.reward_config.get('efficiency_config', {})
        self.efficiency_component = EfficiencyRewardComponent(efficiency_config)
        
        # 平滑性奖励组件
        smoothness_config = self.reward_config.get('smoothness_config', {})
        self.smoothness_component = SmoothnessRewardComponent(smoothness_config)
    
    def calculate_reward(self, 
                        current_position: np.ndarray,
                        target_position: np.ndarray,
                        previous_position: np.ndarray,
                        action: np.ndarray,
                        is_collision: bool = False,
                        is_success: bool = False,
                        is_out_of_bounds: bool = False,
                        current_velocity: Optional[np.ndarray] = None,
                        step: int = 0) -> float:
        """
        计算综合奖励
        
        Args:
            current_position: 当前位置
            target_position: 目标位置
            previous_position: 上一步位置
            action: 执行的动作
            is_collision: 是否碰撞
            is_success: 是否成功到达目标
            is_out_of_bounds: 是否越界
            current_velocity: 当前速度
            step: 当前步数
            
        Returns:
            总奖励值
        """
        total_reward = 0.0
        component_rewards = {}
        
        # 1. 导航奖励
        if self.weights['navigation'] > 0:
            nav_reward, nav_info = self.navigation_component.calculate_reward(
                current_position, target_position, previous_position, step
            )
            weighted_nav_reward = nav_reward * self.weights['navigation']
            total_reward += weighted_nav_reward
            component_rewards['navigation'] = {
                'reward': weighted_nav_reward,
                'raw_reward': nav_reward,
                'info': nav_info
            }
        
        # 2. 安全奖励
        if self.weights['safety'] > 0:
            safety_reward, safety_info = self.safety_component.calculate_reward(
                current_position, is_collision, is_out_of_bounds, step
            )
            weighted_safety_reward = safety_reward * self.weights['safety']
            total_reward += weighted_safety_reward
            component_rewards['safety'] = {
                'reward': weighted_safety_reward,
                'raw_reward': safety_reward,
                'info': safety_info
            }
        
        # 3. 效率奖励
        if self.weights['efficiency'] > 0:
            efficiency_reward, efficiency_info = self.efficiency_component.calculate_reward(
                current_position, target_position, action, step
            )
            weighted_efficiency_reward = efficiency_reward * self.weights['efficiency']
            total_reward += weighted_efficiency_reward
            component_rewards['efficiency'] = {
                'reward': weighted_efficiency_reward,
                'raw_reward': efficiency_reward,
                'info': efficiency_info
            }
        
        # 4. 平滑性奖励
        if self.weights['smoothness'] > 0 and current_velocity is not None:
            smoothness_reward, smoothness_info = self.smoothness_component.calculate_reward(
                action, current_velocity, step
            )
            weighted_smoothness_reward = smoothness_reward * self.weights['smoothness']
            total_reward += weighted_smoothness_reward
            component_rewards['smoothness'] = {
                'reward': weighted_smoothness_reward,
                'raw_reward': smoothness_reward,
                'info': smoothness_info
            }
        
        # 5. 特殊奖励
        special_reward = 0.0
        special_info = {}
        
        # 成功奖励
        if is_success:
            success_bonus = self.success_reward
            special_reward += success_bonus
            special_info['success_reward'] = success_bonus
            self.logger.info(f"Success reward given: {success_bonus}")
        
        # 碰撞惩罚（额外惩罚，不受权重影响）
        if is_collision:
            collision_penalty = self.collision_penalty
            special_reward += collision_penalty
            special_info['collision_penalty'] = collision_penalty
            self.logger.debug(f"Collision penalty: {collision_penalty}")
        
        total_reward += special_reward
        component_rewards['special'] = {
            'reward': special_reward,
            'raw_reward': special_reward,
            'info': special_info
        }
        
        # 记录奖励历史
        self._record_reward_history(total_reward, component_rewards)
        
        # 调试信息
        if step % 100 == 0:  # 每100步记录一次详细信息
            self.logger.debug(f"Step {step} reward breakdown: "
                             f"nav={component_rewards.get('navigation', {}).get('reward', 0):.3f}, "
                             f"safety={component_rewards.get('safety', {}).get('reward', 0):.3f}, "
                             f"eff={component_rewards.get('efficiency', {}).get('reward', 0):.3f}, "
                             f"smooth={component_rewards.get('smoothness', {}).get('reward', 0):.3f}, "
                             f"special={special_reward:.3f}, "
                             f"total={total_reward:.3f}")
        
        return total_reward
    
    def _record_reward_history(self, total_reward: float, component_rewards: Dict[str, Any]):
        """记录奖励历史"""
        self.reward_history.append(total_reward)
        
        for component, reward_data in component_rewards.items():
            if component in self.component_history:
                self.component_history[component].append(reward_data['reward'])
        
        # 限制历史长度
        max_history = 1000
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
            for component in self.component_history:
                if len(self.component_history[component]) > max_history:
                    self.component_history[component] = self.component_history[component][-max_history:]
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        获取奖励统计信息
        
        Returns:
            奖励统计字典
        """
        if not self.reward_history:
            return {'message': 'No reward history available'}
        
        stats = {
            'total_rewards': {
                'mean': np.mean(self.reward_history),
                'std': np.std(self.reward_history),
                'min': np.min(self.reward_history),
                'max': np.max(self.reward_history),
                'count': len(self.reward_history)
            },
            'component_rewards': {}
        }
        
        for component, history in self.component_history.items():
            if history:
                stats['component_rewards'][component] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'contribution_ratio': abs(np.mean(history)) / max(abs(np.mean(self.reward_history)), 1e-6)
                }
        
        return stats
    
    def reset_episode(self):
        """重置回合（清理组件状态）"""
        self.efficiency_component.reset_history()
        self.smoothness_component.reset_history()
        
        self.logger.debug("Reward function reset for new episode")
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        动态更新奖励权重
        
        Args:
            new_weights: 新的权重字典
        """
        for component, weight in new_weights.items():
            if component in self.weights:
                old_weight = self.weights[component]
                self.weights[component] = weight
                self.logger.info(f"Updated {component} weight: {old_weight} -> {weight}")
    
    def get_shaped_reward(self, 
                         current_position: np.ndarray,
                         target_position: np.ndarray,
                         step: int,
                         max_steps: int) -> float:
        """
        获取形状奖励（用于引导学习）
        
        Args:
            current_position: 当前位置
            target_position: 目标位置
            step: 当前步数
            max_steps: 最大步数
            
        Returns:
            形状奖励
        """
        # 基于距离的形状奖励
        distance = np.linalg.norm(current_position - target_position)
        distance_reward = -distance / 50.0  # 基础距离奖励
        
        # 时间衰减因子（鼓励快速完成）
        time_factor = 1.0 - step / max_steps
        
        # 组合形状奖励
        shaped_reward = distance_reward * time_factor
        
        return shaped_reward
    
    def get_curriculum_reward_weights(self, 
                                    episode: int, 
                                    success_rate: float) -> Dict[str, float]:
        """
        获取课程学习的动态奖励权重
        
        Args:
            episode: 当前回合数
            success_rate: 成功率
            
        Returns:
            动态权重字典
        """
        # 基础权重
        base_weights = self.weights.copy()
        
        # 根据训练进度调整权重
        if episode < 1000:
            # 早期训练：重视安全和基础导航
            base_weights['safety'] *= 1.5
            base_weights['navigation'] *= 1.2
            base_weights['efficiency'] *= 0.5
            base_weights['smoothness'] *= 0.3
            
        elif success_rate > 0.7:
            # 高成功率时：重视效率和平滑性
            base_weights['efficiency'] *= 1.5
            base_weights['smoothness'] *= 2.0
            base_weights['navigation'] *= 0.8
            
        elif success_rate < 0.3:
            # 低成功率时：重视导航和安全
            base_weights['navigation'] *= 1.5
            base_weights['safety'] *= 1.3
            base_weights['efficiency'] *= 0.7
        
        return base_weights
    
    def visualize_reward_breakdown(self) -> Dict[str, List[float]]:
        """
        获取奖励分解可视化数据
        
        Returns:
            各组件奖励历史
        """
        return {
            component: history.copy() 
            for component, history in self.component_history.items()
            if history
        }