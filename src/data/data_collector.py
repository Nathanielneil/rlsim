"""
数据采集器 - 收集训练和评估过程中的各种数据
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, deque

from ..utils.logger import get_logger
from ..utils.file_manager import FileManager


@dataclass
class StepData:
    """单步数据结构"""
    step: int
    timestamp: float
    observation: Any
    action: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]
    
    # 环境状态
    position: np.ndarray
    velocity: np.ndarray
    target_position: np.ndarray
    distance_to_target: float
    
    # 奖励分解
    reward_components: Optional[Dict[str, float]] = None


@dataclass
class EpisodeData:
    """回合数据结构"""
    episode: int
    start_time: float
    end_time: float
    total_reward: float
    episode_length: int
    success: bool
    termination_reason: str
    
    # 轨迹信息
    trajectory: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    
    # 统计指标
    trajectory_length: float
    navigation_error: float
    collision_count: int
    
    # 性能指标
    oracle_success: bool
    spl: float  # Success weighted by Path Length


class DataCollector:
    """数据采集器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据采集器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = get_logger("data_collector")
        self.file_manager = FileManager()
        
        # 采集配置
        self.collect_detailed_data = self.config.get('collect_detailed_data', True)
        self.save_trajectories = self.config.get('save_trajectories', True)
        self.max_episodes_in_memory = self.config.get('max_episodes_in_memory', 1000)
        
        # 数据存储
        self.current_episode_data = []
        self.episodes_data = deque(maxlen=self.max_episodes_in_memory)
        self.training_statistics = defaultdict(list)
        
        # 当前回合状态
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = None
        self.episode_start_position = None
        self.episode_target_position = None
        
        # 全局统计
        self.global_step_count = 0
        self.total_episodes = 0
        self.collection_start_time = time.time()
        
        self.logger.info("Data collector initialized")
    
    def start_episode(self, episode: int, start_position: np.ndarray, target_position: np.ndarray):
        """
        开始新回合的数据采集
        
        Args:
            episode: 回合编号
            start_position: 起始位置
            target_position: 目标位置
        """
        self.current_episode = episode
        self.current_step = 0
        self.episode_start_time = time.time()
        self.episode_start_position = start_position.copy()
        self.episode_target_position = target_position.copy()
        
        # 清空当前回合数据
        self.current_episode_data = []
        
        self.logger.debug(f"Started episode {episode} data collection")
    
    def collect_step(self, 
                    observation: Any,
                    action: np.ndarray,
                    reward: float,
                    done: bool,
                    truncated: bool,
                    info: Dict[str, Any],
                    reward_components: Optional[Dict[str, float]] = None):
        """
        收集单步数据
        
        Args:
            observation: 观测
            action: 动作
            reward: 奖励
            done: 是否完成
            truncated: 是否截断
            info: 信息字典
            reward_components: 奖励组件分解
        """
        current_time = time.time()
        
        # 提取环境状态信息
        position = info.get('current_position', np.zeros(3))
        target_position = info.get('target_position', self.episode_target_position)
        distance_to_target = info.get('distance_to_target', np.linalg.norm(position - target_position))
        
        # 估计速度（如果没有直接提供）
        if len(self.current_episode_data) > 0:
            prev_position = self.current_episode_data[-1].position
            dt = current_time - self.current_episode_data[-1].timestamp
            velocity = (position - prev_position) / max(dt, 0.01)
        else:
            velocity = np.zeros(3)
        
        # 创建步数据
        step_data = StepData(
            step=self.current_step,
            timestamp=current_time,
            observation=observation if self.collect_detailed_data else None,
            action=action.copy(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info.copy(),
            position=position.copy(),
            velocity=velocity.copy(),
            target_position=target_position.copy(),
            distance_to_target=distance_to_target,
            reward_components=reward_components.copy() if reward_components else None
        )
        
        self.current_episode_data.append(step_data)
        self.current_step += 1
        self.global_step_count += 1
    
    def end_episode(self, success: bool, termination_reason: str) -> EpisodeData:
        """
        结束回合并计算统计数据
        
        Args:
            success: 是否成功
            termination_reason: 终止原因
            
        Returns:
            回合数据
        """
        episode_end_time = time.time()
        
        if not self.current_episode_data:
            self.logger.warning(f"No data collected for episode {self.current_episode}")
            return None
        
        # 计算基础统计
        total_reward = sum(step.reward for step in self.current_episode_data)
        episode_length = len(self.current_episode_data)
        
        # 提取轨迹和动作
        trajectory = [step.position for step in self.current_episode_data]
        actions = [step.action for step in self.current_episode_data]
        rewards = [step.reward for step in self.current_episode_data]
        
        # 计算轨迹长度
        trajectory_length = 0.0
        for i in range(1, len(trajectory)):
            trajectory_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        # 计算导航误差
        final_position = trajectory[-1]
        navigation_error = np.linalg.norm(final_position - self.episode_target_position)
        
        # 计算碰撞次数
        collision_count = sum(1 for step in self.current_episode_data 
                            if step.info.get('collision', False))
        
        # 计算Oracle成功率（是否曾经接近过目标）
        min_distance = min(step.distance_to_target for step in self.current_episode_data)
        success_threshold = 3.0  # 成功阈值
        oracle_success = min_distance <= success_threshold
        
        # 计算SPL（Success weighted by Path Length）
        optimal_length = np.linalg.norm(self.episode_target_position - self.episode_start_position)
        spl = 0.0
        if success and trajectory_length > 0:
            spl = optimal_length / max(optimal_length, trajectory_length)
        
        # 创建回合数据
        episode_data = EpisodeData(
            episode=self.current_episode,
            start_time=self.episode_start_time,
            end_time=episode_end_time,
            total_reward=total_reward,
            episode_length=episode_length,
            success=success,
            termination_reason=termination_reason,
            trajectory=trajectory,
            actions=actions,
            rewards=rewards,
            trajectory_length=trajectory_length,
            navigation_error=navigation_error,
            collision_count=collision_count,
            oracle_success=oracle_success,
            spl=spl
        )
        
        # 存储回合数据
        self.episodes_data.append(episode_data)
        self.total_episodes += 1
        
        # 更新训练统计
        self._update_training_statistics(episode_data)
        
        self.logger.debug(f"Episode {self.current_episode} completed - "
                         f"Reward: {total_reward:.2f}, Success: {success}, "
                         f"Length: {trajectory_length:.2f}m")
        
        return episode_data
    
    def _update_training_statistics(self, episode_data: EpisodeData):
        """更新训练统计数据"""
        stats = self.training_statistics
        
        # 基础指标
        stats['episode_rewards'].append(episode_data.total_reward)
        stats['episode_lengths'].append(episode_data.episode_length)
        stats['trajectory_lengths'].append(episode_data.trajectory_length)
        stats['navigation_errors'].append(episode_data.navigation_error)
        stats['success_rates'].append(episode_data.success)
        stats['oracle_success_rates'].append(episode_data.oracle_success)
        stats['spl_values'].append(episode_data.spl)
        stats['collision_counts'].append(episode_data.collision_count)
        
        # 时间统计
        episode_duration = episode_data.end_time - episode_data.start_time
        stats['episode_durations'].append(episode_duration)
        
        # 奖励组件统计（如果有的话）
        if self.current_episode_data and self.current_episode_data[0].reward_components:
            for component, values in self._extract_reward_components().items():
                stats[f'reward_{component}'].extend(values)
    
    def _extract_reward_components(self) -> Dict[str, List[float]]:
        """提取奖励组件数据"""
        components = defaultdict(list)
        
        for step in self.current_episode_data:
            if step.reward_components:
                for component, value in step.reward_components.items():
                    components[component].append(value)
        
        return dict(components)
    
    def get_recent_statistics(self, window_size: int = 100) -> Dict[str, Any]:
        """
        获取最近N个回合的统计数据
        
        Args:
            window_size: 窗口大小
            
        Returns:
            统计数据字典
        """
        if not self.episodes_data:
            return {'message': 'No episode data available'}
        
        # 获取最近的回合
        recent_episodes = list(self.episodes_data)[-window_size:]
        
        stats = {
            'num_episodes': len(recent_episodes),
            'mean_reward': np.mean([ep.total_reward for ep in recent_episodes]),
            'std_reward': np.std([ep.total_reward for ep in recent_episodes]),
            'mean_length': np.mean([ep.episode_length for ep in recent_episodes]),
            'mean_trajectory_length': np.mean([ep.trajectory_length for ep in recent_episodes]),
            'mean_navigation_error': np.mean([ep.navigation_error for ep in recent_episodes]),
            'success_rate': np.mean([ep.success for ep in recent_episodes]),
            'oracle_success_rate': np.mean([ep.oracle_success for ep in recent_episodes]),
            'mean_spl': np.mean([ep.spl for ep in recent_episodes]),
            'collision_rate': np.mean([min(ep.collision_count, 1) for ep in recent_episodes]),
            'mean_collision_count': np.mean([ep.collision_count for ep in recent_episodes])
        }
        
        # 添加中位数和极值
        rewards = [ep.total_reward for ep in recent_episodes]
        stats.update({
            'median_reward': np.median(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        })
        
        return stats
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        获取全局统计数据
        
        Returns:
            全局统计字典
        """
        if not self.episodes_data:
            return {'message': 'No episode data available'}
        
        all_episodes = list(self.episodes_data)
        collection_time = time.time() - self.collection_start_time
        
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.global_step_count,
            'collection_time': collection_time,
            'episodes_per_hour': self.total_episodes / (collection_time / 3600),
            'steps_per_second': self.global_step_count / collection_time,
            
            # 性能指标
            'overall_success_rate': np.mean([ep.success for ep in all_episodes]),
            'overall_oracle_success_rate': np.mean([ep.oracle_success for ep in all_episodes]),
            'mean_spl': np.mean([ep.spl for ep in all_episodes]),
            'mean_navigation_error': np.mean([ep.navigation_error for ep in all_episodes]),
            'mean_trajectory_length': np.mean([ep.trajectory_length for ep in all_episodes]),
            
            # 奖励统计
            'mean_reward': np.mean([ep.total_reward for ep in all_episodes]),
            'std_reward': np.std([ep.total_reward for ep in all_episodes]),
            'best_reward': np.max([ep.total_reward for ep in all_episodes]),
            'worst_reward': np.min([ep.total_reward for ep in all_episodes]),
            
            # 效率统计
            'mean_episode_length': np.mean([ep.episode_length for ep in all_episodes]),
            'mean_episode_duration': np.mean([ep.end_time - ep.start_time for ep in all_episodes]),
            
            # 安全统计
            'collision_rate': np.mean([min(ep.collision_count, 1) for ep in all_episodes]),
            'mean_collision_count': np.mean([ep.collision_count for ep in all_episodes])
        }
        
        return stats
    
    def save_episode_data(self, episode_data: EpisodeData, save_detailed: bool = False) -> str:
        """
        保存回合数据
        
        Args:
            episode_data: 回合数据
            save_detailed: 是否保存详细数据
            
        Returns:
            保存的文件路径
        """
        if not self.save_trajectories:
            return None
        
        # 准备保存数据
        save_data = {
            'episode': episode_data.episode,
            'start_time': episode_data.start_time,
            'end_time': episode_data.end_time,
            'total_reward': episode_data.total_reward,
            'episode_length': episode_data.episode_length,
            'success': episode_data.success,
            'termination_reason': episode_data.termination_reason,
            'trajectory_length': episode_data.trajectory_length,
            'navigation_error': episode_data.navigation_error,
            'collision_count': episode_data.collision_count,
            'oracle_success': episode_data.oracle_success,
            'spl': episode_data.spl,
            'trajectory': [pos.tolist() for pos in episode_data.trajectory],
            'actions': [action.tolist() for action in episode_data.actions],
            'rewards': episode_data.rewards
        }
        
        if save_detailed and self.current_episode_data:
            # 保存详细步骤数据
            detailed_steps = []
            for step in self.current_episode_data:
                step_dict = {
                    'step': step.step,
                    'timestamp': step.timestamp,
                    'action': step.action.tolist(),
                    'reward': step.reward,
                    'done': step.done,
                    'position': step.position.tolist(),
                    'velocity': step.velocity.tolist(),
                    'distance_to_target': step.distance_to_target
                }
                
                if step.reward_components:
                    step_dict['reward_components'] = step.reward_components
                
                detailed_steps.append(step_dict)
            
            save_data['detailed_steps'] = detailed_steps
        
        # 保存文件
        filename = f"episode_{episode_data.episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = self.file_manager.save_trajectory(save_data, filename)
        
        return filepath
    
    def export_statistics(self, filename: Optional[str] = None) -> str:
        """
        导出统计数据
        
        Args:
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"training_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 准备导出数据
        export_data = {
            'global_statistics': self.get_global_statistics(),
            'recent_statistics': self.get_recent_statistics(),
            'training_progress': {
                key: values[-1000:] for key, values in self.training_statistics.items()
            },  # 只保存最近1000个数据点
            'collection_metadata': {
                'collection_start_time': self.collection_start_time,
                'total_episodes': self.total_episodes,
                'global_step_count': self.global_step_count,
                'config': self.config
            }
        }
        
        filepath = self.file_manager.save_results(export_data, filename)
        self.logger.info(f"Statistics exported to {filepath}")
        
        return filepath
    
    def clear_old_data(self, keep_episodes: int = 100):
        """
        清理旧数据以节省内存
        
        Args:
            keep_episodes: 保留的回合数
        """
        if len(self.episodes_data) > keep_episodes:
            # 保留最近的回合
            episodes_to_keep = list(self.episodes_data)[-keep_episodes:]
            self.episodes_data.clear()
            self.episodes_data.extend(episodes_to_keep)
            
            # 清理统计数据
            for key, values in self.training_statistics.items():
                if len(values) > keep_episodes:
                    self.training_statistics[key] = values[-keep_episodes:]
            
            self.logger.info(f"Cleared old data, kept {keep_episodes} episodes")
    
    def reset(self):
        """重置数据采集器"""
        self.current_episode_data = []
        self.episodes_data.clear()
        self.training_statistics.clear()
        
        self.current_episode = 0
        self.current_step = 0
        self.global_step_count = 0
        self.total_episodes = 0
        self.collection_start_time = time.time()
        
        self.logger.info("Data collector reset")