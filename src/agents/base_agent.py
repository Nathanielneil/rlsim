"""
基础智能体类 - 为所有强化学习算法提供统一接口
"""

import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import time
from datetime import datetime

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger
from ..utils.file_manager import FileManager


class BaseAgent(ABC):
    """基础智能体抽象类"""
    
    def __init__(self, 
                 env,
                 algorithm_name: str,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化基础智能体
        
        Args:
            env: 环境实例
            algorithm_name: 算法名称
            config: 配置字典
            device: 计算设备
        """
        self.env = env
        self.algorithm_name = algorithm_name.upper()
        self.config = config or {}
        
        # 设置计算设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 初始化工具
        self.logger = get_logger(f"{algorithm_name.lower()}_agent")
        self.file_manager = FileManager()
        
        # 获取环境信息
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # 训练参数
        self.total_timesteps = self.config.get('algorithm_params', {}).get('total_timesteps', 1000000)
        self.save_freq = self.config.get('training', {}).get('save_freq', 10000)
        self.eval_freq = self.config.get('training', {}).get('eval_freq', 5000)
        self.log_interval = self.config.get('training', {}).get('log_interval', 100)
        
        # 训练状态
        self.global_step = 0
        self.episode_count = 0
        self.total_episodes = 0
        self.training_start_time = None
        
        # 性能统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.evaluation_results = []
        
        # 网络模型（子类实现）
        self.policy_net = None
        self.value_net = None
        self.target_net = None
        
        self.logger.info(f"{self.algorithm_name} agent initialized on {self.device}")
    
    @abstractmethod
    def build_networks(self) -> None:
        """构建神经网络（子类必须实现）"""
        pass
    
    @abstractmethod
    def select_action(self, observation: Union[Dict[str, Any], np.ndarray], 
                     deterministic: bool = False) -> np.ndarray:
        """
        选择动作（子类必须实现）
        
        Args:
            observation: 观测
            deterministic: 是否使用确定性策略
            
        Returns:
            动作
        """
        pass
    
    @abstractmethod
    def update(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        更新网络参数（子类必须实现）
        
        Args:
            batch_data: 批次数据
            
        Returns:
            损失字典
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str, episode: int) -> None:
        """
        保存模型（子类必须实现）
        
        Args:
            filepath: 保存路径
            episode: 回合数
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        加载模型（子类必须实现）
        
        Args:
            filepath: 模型路径
        """
        pass
    
    def train(self, total_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        训练智能体
        
        Args:
            total_episodes: 总回合数（可选）
            
        Returns:
            训练结果字典
        """
        self.training_start_time = time.time()
        
        if total_episodes is not None:
            self.total_episodes = total_episodes
        
        self.logger.info(f"Starting {self.algorithm_name} training for {self.total_episodes} episodes")
        
        # 训练主循环
        try:
            for episode in range(self.total_episodes):
                episode_result = self._train_episode(episode)
                
                # 记录结果
                self.episode_rewards.append(episode_result['total_reward'])
                self.episode_lengths.append(episode_result['episode_length'])
                
                # 定期保存和评估
                if episode > 0:
                    if episode % (self.save_freq // 100) == 0:  # 假设平均每回合100步
                        self._save_checkpoint(episode)
                    
                    if episode % (self.eval_freq // 100) == 0:
                        eval_result = self._evaluate_agent()
                        self.evaluation_results.append(eval_result)
                    
                    if episode % self.log_interval == 0:
                        self._log_training_progress(episode)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        # 训练完成
        training_time = time.time() - self.training_start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # 保存最终模型
        final_model_path = self._save_checkpoint(self.total_episodes, suffix='final')
        
        # 返回训练结果
        training_results = {
            'algorithm': self.algorithm_name,
            'total_episodes': self.total_episodes,
            'training_time': training_time,
            'final_model_path': final_model_path,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_results': self.evaluation_results,
            'success_rate_history': self.success_rate_history
        }
        
        return training_results
    
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """
        训练单个回合
        
        Args:
            episode: 回合索引
            
        Returns:
            回合结果字典
        """
        observation, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False
        
        episode_data = []
        
        while not (done or truncated):
            # 选择动作
            action = self.select_action(observation, deterministic=False)
            
            # 执行动作
            next_observation, reward, done, truncated, info = self.env.step(action)
            
            # 存储经验
            episode_data.append({
                'observation': observation,
                'action': action,
                'reward': reward,
                'next_observation': next_observation,
                'done': done or truncated,
                'info': info
            })
            
            # 更新状态
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
            
            # 在线更新（如果支持）
            if hasattr(self, '_should_update') and self._should_update():
                batch_data = self._prepare_batch_data(episode_data)
                if batch_data is not None:
                    update_info = self.update(batch_data)
                    if self.global_step % self.log_interval == 0:
                        self._log_update_info(update_info)
        
        # 回合结束处理
        success = info.get('success', False)
        if success:
            self.logger.info(f"Episode {episode} SUCCESS - Reward: {episode_reward:.2f}, Length: {episode_length}")
        
        # 批量更新（如果支持）
        if hasattr(self, '_update_at_episode_end'):
            batch_data = self._prepare_episode_batch_data(episode_data)
            if batch_data is not None:
                update_info = self.update(batch_data)
                self._log_update_info(update_info)
        
        return {
            'episode': episode,
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'success': success,
            'episode_data': episode_data
        }
    
    def _evaluate_agent(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            评估结果字典
        """
        self.logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        for _ in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(observation, deterministic=True)
                observation, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(info.get('success', False))
        
        # 计算统计数据
        eval_result = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'num_episodes': num_episodes
        }
        
        self.success_rate_history.append(eval_result['success_rate'])
        
        self.logger.info(f"Evaluation results - Mean reward: {eval_result['mean_reward']:.2f}, "
                        f"Success rate: {eval_result['success_rate']:.2%}")
        
        return eval_result
    
    def _save_checkpoint(self, episode: int, suffix: str = '') -> str:
        """
        保存检查点
        
        Args:
            episode: 回合数
            suffix: 文件名后缀
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if suffix:
            filename = f"{timestamp}_{self.algorithm_name.lower()}_{episode}_{suffix}.pth"
        else:
            filename = f"{timestamp}_{self.algorithm_name.lower()}_{episode}.pth"
        
        # 准备保存的元数据
        metadata = {
            'episode': episode,
            'global_step': self.global_step,
            'algorithm': self.algorithm_name,
            'config': self.config,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'episode_rewards': self.episode_rewards[-100:],  # 只保存最近100个
            'success_rate_history': self.success_rate_history,
            'device': str(self.device)
        }
        
        # 保存模型
        model_path = self.file_manager.save_model(
            self,
            algorithm=self.algorithm_name.lower(),
            episode=episode,
            metadata=metadata,
            timestamp=timestamp
        )
        
        self.logger.info(f"Model saved: {model_path}")
        return model_path
    
    def _log_training_progress(self, episode: int):
        """记录训练进度"""
        if not self.episode_rewards:
            return
        
        # 最近100个回合的统计
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        
        # 成功率（如果有评估结果）
        success_rate = self.success_rate_history[-1] if self.success_rate_history else 0.0
        
        # 训练时间
        elapsed_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        self.logger.info(f"Episode {episode}/{self.total_episodes} - "
                        f"Avg Reward: {avg_reward:.2f}, "
                        f"Avg Length: {avg_length:.1f}, "
                        f"Success Rate: {success_rate:.2%}, "
                        f"Time: {elapsed_time:.1f}s")
    
    def _log_update_info(self, update_info: Dict[str, float]):
        """记录更新信息"""
        if update_info:
            info_str = ", ".join([f"{k}: {v:.4f}" for k, v in update_info.items()])
            self.logger.debug(f"Update info - {info_str}")
    
    def _prepare_batch_data(self, episode_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        准备批次数据（子类可重写）
        
        Args:
            episode_data: 回合数据
            
        Returns:
            批次数据或None
        """
        return None
    
    def _prepare_episode_batch_data(self, episode_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        准备回合批次数据（子类可重写）
        
        Args:
            episode_data: 回合数据
            
        Returns:
            批次数据或None
        """
        return None
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            统计信息字典
        """
        if not self.episode_rewards:
            return {'message': 'No training data available'}
        
        stats = {
            'algorithm': self.algorithm_name,
            'total_episodes': len(self.episode_rewards),
            'global_steps': self.global_step,
            'rewards': {
                'mean': np.mean(self.episode_rewards),
                'std': np.std(self.episode_rewards),
                'min': np.min(self.episode_rewards),
                'max': np.max(self.episode_rewards),
                'recent_mean': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            },
            'episode_lengths': {
                'mean': np.mean(self.episode_lengths),
                'std': np.std(self.episode_lengths),
                'min': np.min(self.episode_lengths),
                'max': np.max(self.episode_lengths)
            },
            'success_rates': self.success_rate_history,
            'current_success_rate': self.success_rate_history[-1] if self.success_rate_history else 0.0
        }
        
        if self.training_start_time:
            stats['training_time'] = time.time() - self.training_start_time
        
        return stats
    
    def set_eval_mode(self):
        """设置评估模式"""
        if hasattr(self, 'policy_net') and self.policy_net is not None:
            self.policy_net.eval()
        if hasattr(self, 'value_net') and self.value_net is not None:
            self.value_net.eval()
    
    def set_train_mode(self):
        """设置训练模式"""
        if hasattr(self, 'policy_net') and self.policy_net is not None:
            self.policy_net.train()
        if hasattr(self, 'value_net') and self.value_net is not None:
            self.value_net.train()
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("Cleaning up agent resources")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存最终统计
        if self.episode_rewards:
            stats = self.get_training_statistics()
            stats_filename = f"{self.algorithm_name.lower()}_final_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_results(stats, stats_filename)
    
    def __del__(self):
        """析构函数"""
        self.cleanup()