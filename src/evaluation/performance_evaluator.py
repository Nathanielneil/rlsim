"""
性能评估器 - 全面评估无人机导航算法的性能
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from ..utils.logger import get_logger
from ..utils.file_manager import FileManager
from ..data.data_collector import DataCollector, EpisodeData


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # 基础指标
    success_rate: float
    oracle_success_rate: float
    navigation_error: float
    trajectory_length: float
    spl: float  # Success weighted by Path Length
    
    # 碰撞指标
    navigation_collision: float  # N-C
    collision_rate: float
    
    # 效率指标
    mean_episode_length: float
    mean_episode_time: float
    
    # 稳定性指标
    velocity_smoothness: float
    action_smoothness: float
    
    # 统计指标
    reward_mean: float
    reward_std: float
    num_episodes: int


class PerformanceEvaluator:
    """性能评估器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or {}
        self.logger = get_logger("performance_evaluator")
        self.file_manager = FileManager()
        
        # 评估参数
        self.success_threshold = self.config.get('success_threshold', 3.0)
        self.oracle_threshold = self.config.get('oracle_threshold', 3.0)
        self.collision_threshold = self.config.get('collision_threshold', 0.5)
        
        # 评估历史
        self.evaluation_history = []
        
        self.logger.info("Performance evaluator initialized")
    
    def evaluate_agent(self, 
                      agent, 
                      env, 
                      num_episodes: int = 100,
                      deterministic: bool = True,
                      save_trajectories: bool = False) -> EvaluationMetrics:
        """
        评估智能体性能
        
        Args:
            agent: 智能体实例
            env: 环境实例
            num_episodes: 评估回合数
            deterministic: 是否使用确定性策略
            save_trajectories: 是否保存轨迹数据
            
        Returns:
            评估指标
        """
        self.logger.info(f"Starting agent evaluation for {num_episodes} episodes")
        
        # 初始化数据收集器
        collector = DataCollector({
            'collect_detailed_data': save_trajectories,
            'save_trajectories': save_trajectories
        })
        
        # 设置智能体为评估模式
        agent.set_eval_mode()
        
        evaluation_start_time = time.time()
        episode_results = []
        
        try:
            for episode in range(num_episodes):
                episode_result = self._evaluate_single_episode(
                    agent, env, collector, episode, deterministic
                )
                episode_results.append(episode_result)
                
                # 定期记录进度
                if (episode + 1) % max(1, num_episodes // 10) == 0:
                    progress = (episode + 1) / num_episodes * 100
                    self.logger.info(f"Evaluation progress: {progress:.1f}% ({episode + 1}/{num_episodes})")
        
        except KeyboardInterrupt:
            self.logger.warning("Evaluation interrupted by user")
            episode_results = episode_results[:episode]
        
        evaluation_time = time.time() - evaluation_start_time
        
        # 计算评估指标
        metrics = self._calculate_metrics(episode_results, evaluation_time)
        
        # 记录评估历史
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': len(episode_results),
            'evaluation_time': evaluation_time,
            'metrics': metrics,
            'config': self.config
        }
        self.evaluation_history.append(evaluation_record)
        
        self.logger.info(f"Evaluation completed - Success Rate: {metrics.success_rate:.2%}, "
                        f"Mean Navigation Error: {metrics.navigation_error:.2f}m, "
                        f"Mean SPL: {metrics.spl:.3f}")
        
        return metrics
    
    def _evaluate_single_episode(self, 
                                agent, 
                                env, 
                                collector: DataCollector, 
                                episode: int, 
                                deterministic: bool) -> EpisodeData:
        """评估单个回合"""
        observation, info = env.reset()
        
        # 获取起始和目标位置
        start_position = info.get('current_position', np.array([0, 0, -2]))
        target_position = info.get('target_position', np.array([10, 10, -5]))
        
        collector.start_episode(episode, start_position, target_position)
        
        episode_reward = 0.0
        step_count = 0
        done = False
        truncated = False
        
        # 存储步骤数据用于平滑性分析
        velocities = []
        actions = []
        
        while not (done or truncated):
            # 选择动作
            action = agent.select_action(observation, deterministic=deterministic)
            actions.append(action.copy())
            
            # 执行动作
            next_observation, reward, done, truncated, info = env.step(action)
            
            # 提取速度信息（如果可用）
            if 'velocity' in info:
                velocities.append(info['velocity'])
            
            # 收集步骤数据
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                truncated=truncated,
                info=info
            )
            
            observation = next_observation
            episode_reward += reward
            step_count += 1
        
        # 确定成功状态和终止原因
        success = info.get('success', False)
        termination_reason = info.get('termination_reason', 'unknown')
        
        # 结束回合数据收集
        episode_data = collector.end_episode(success, termination_reason)
        
        # 添加平滑性分析数据
        if velocities and len(velocities) > 1:
            episode_data.velocity_smoothness = self._calculate_velocity_smoothness(velocities)
        else:
            episode_data.velocity_smoothness = 0.0
        
        if len(actions) > 1:
            episode_data.action_smoothness = self._calculate_action_smoothness(actions)
        else:
            episode_data.action_smoothness = 0.0
        
        return episode_data
    
    def _calculate_velocity_smoothness(self, velocities: List[np.ndarray]) -> float:
        """计算速度平滑性"""
        if len(velocities) < 2:
            return 0.0
        
        # 计算速度变化的方差
        velocity_changes = []
        for i in range(1, len(velocities)):
            velocity_change = np.linalg.norm(velocities[i] - velocities[i-1])
            velocity_changes.append(velocity_change)
        
        if not velocity_changes:
            return 0.0
        
        # 平滑性 = 1 / (1 + std(velocity_changes))
        smoothness = 1.0 / (1.0 + np.std(velocity_changes))
        return smoothness
    
    def _calculate_action_smoothness(self, actions: List[np.ndarray]) -> float:
        """计算动作平滑性"""
        if len(actions) < 2:
            return 0.0
        
        # 计算动作变化的方差
        action_changes = []
        for i in range(1, len(actions)):
            action_change = np.linalg.norm(actions[i] - actions[i-1])
            action_changes.append(action_change)
        
        if not action_changes:
            return 0.0
        
        # 平滑性 = 1 / (1 + std(action_changes))
        smoothness = 1.0 / (1.0 + np.std(action_changes))
        return smoothness
    
    def _calculate_metrics(self, episode_results: List[EpisodeData], evaluation_time: float) -> EvaluationMetrics:
        """计算评估指标"""
        if not episode_results:
            self.logger.warning("No episode results to calculate metrics")
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # 基础指标
        success_rate = np.mean([ep.success for ep in episode_results])
        oracle_success_rate = np.mean([ep.oracle_success for ep in episode_results])
        navigation_error = np.mean([ep.navigation_error for ep in episode_results])
        trajectory_length = np.mean([ep.trajectory_length for ep in episode_results])
        
        # SPL指标
        spl_values = [ep.spl for ep in episode_results if ep.spl > 0]
        spl = np.mean(spl_values) if spl_values else 0.0
        
        # 碰撞指标
        collision_episodes = [ep for ep in episode_results if ep.collision_count > 0]
        collision_rate = len(collision_episodes) / len(episode_results)
        
        # 导航碰撞率（N-C）：整个导航过程中碰撞时间的平均比率
        navigation_collision = 0.0
        total_steps = 0
        collision_steps = 0
        for ep in episode_results:
            total_steps += ep.episode_length
            collision_steps += ep.collision_count
        
        if total_steps > 0:
            navigation_collision = collision_steps / total_steps
        
        # 效率指标
        mean_episode_length = np.mean([ep.episode_length for ep in episode_results])
        episode_durations = [ep.end_time - ep.start_time for ep in episode_results]
        mean_episode_time = np.mean(episode_durations)
        
        # 稳定性指标
        velocity_smoothness_values = [getattr(ep, 'velocity_smoothness', 0) for ep in episode_results]
        action_smoothness_values = [getattr(ep, 'action_smoothness', 0) for ep in episode_results]
        velocity_smoothness = np.mean(velocity_smoothness_values)
        action_smoothness = np.mean(action_smoothness_values)
        
        # 奖励统计
        rewards = [ep.total_reward for ep in episode_results]
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        
        return EvaluationMetrics(
            success_rate=success_rate,
            oracle_success_rate=oracle_success_rate,
            navigation_error=navigation_error,
            trajectory_length=trajectory_length,
            spl=spl,
            navigation_collision=navigation_collision,
            collision_rate=collision_rate,
            mean_episode_length=mean_episode_length,
            mean_episode_time=mean_episode_time,
            velocity_smoothness=velocity_smoothness,
            action_smoothness=action_smoothness,
            reward_mean=reward_mean,
            reward_std=reward_std,
            num_episodes=len(episode_results)
        )
    
    def compare_algorithms(self, 
                          evaluations: Dict[str, EvaluationMetrics], 
                          save_report: bool = True) -> Dict[str, Any]:
        """
        比较多个算法的性能
        
        Args:
            evaluations: 算法名称到评估指标的映射
            save_report: 是否保存比较报告
            
        Returns:
            比较结果字典
        """
        self.logger.info(f"Comparing {len(evaluations)} algorithms")
        
        if not evaluations:
            return {'error': 'No evaluations provided'}
        
        # 创建比较表
        comparison = {}
        
        # 所有指标名称
        metric_names = [
            'success_rate', 'oracle_success_rate', 'navigation_error', 
            'trajectory_length', 'spl', 'navigation_collision', 'collision_rate',
            'mean_episode_length', 'mean_episode_time', 'velocity_smoothness', 
            'action_smoothness', 'reward_mean', 'reward_std'
        ]
        
        # 为每个指标创建比较
        for metric in metric_names:
            comparison[metric] = {}
            values = {}
            
            for algo_name, metrics in evaluations.items():
                value = getattr(metrics, metric)
                values[algo_name] = value
                comparison[metric][algo_name] = value
            
            # 找出最佳算法
            if metric in ['success_rate', 'oracle_success_rate', 'spl', 
                         'velocity_smoothness', 'action_smoothness', 'reward_mean']:
                # 越大越好的指标
                best_algo = max(values, key=values.get)
            else:
                # 越小越好的指标
                best_algo = min(values, key=values.get)
            
            comparison[metric]['best_algorithm'] = best_algo
            comparison[metric]['best_value'] = values[best_algo]
        
        # 综合排名
        ranking_scores = defaultdict(float)
        
        for metric in metric_names:
            values = {algo: getattr(evaluations[algo], metric) 
                     for algo in evaluations.keys()}
            
            # 标准化分数（0-1之间）
            min_val = min(values.values())
            max_val = max(values.values())
            
            if max_val != min_val:
                for algo in values:
                    if metric in ['success_rate', 'oracle_success_rate', 'spl',
                                 'velocity_smoothness', 'action_smoothness', 'reward_mean']:
                        # 越大越好
                        normalized_score = (values[algo] - min_val) / (max_val - min_val)
                    else:
                        # 越小越好
                        normalized_score = 1.0 - (values[algo] - min_val) / (max_val - min_val)
                    
                    # 根据指标重要性加权
                    weight = self._get_metric_weight(metric)
                    ranking_scores[algo] += normalized_score * weight
        
        # 排序
        sorted_algorithms = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison['overall_ranking'] = [
            {'algorithm': algo, 'score': score} 
            for algo, score in sorted_algorithms
        ]
        
        # 添加元数据
        comparison['metadata'] = {
            'comparison_time': datetime.now().isoformat(),
            'num_algorithms': len(evaluations),
            'evaluator_config': self.config
        }
        
        # 保存报告
        if save_report:
            self._save_comparison_report(comparison, evaluations)
        
        self.logger.info(f"Algorithm comparison completed. Best overall: {sorted_algorithms[0][0]}")
        
        return comparison
    
    def _get_metric_weight(self, metric: str) -> float:
        """获取指标权重"""
        weights = {
            'success_rate': 3.0,           # 最重要
            'oracle_success_rate': 2.0,
            'navigation_error': 2.5,
            'spl': 2.0,
            'collision_rate': 2.5,
            'navigation_collision': 2.0,
            'trajectory_length': 1.5,
            'mean_episode_length': 1.0,
            'mean_episode_time': 1.0,
            'velocity_smoothness': 1.5,
            'action_smoothness': 1.5,
            'reward_mean': 1.0,
            'reward_std': 0.5              # 最不重要
        }
        
        return weights.get(metric, 1.0)
    
    def _save_comparison_report(self, comparison: Dict[str, Any], evaluations: Dict[str, EvaluationMetrics]):
        """保存比较报告"""
        report_data = {
            'comparison': comparison,
            'detailed_evaluations': {
                algo_name: {
                    'success_rate': metrics.success_rate,
                    'oracle_success_rate': metrics.oracle_success_rate,
                    'navigation_error': metrics.navigation_error,
                    'trajectory_length': metrics.trajectory_length,
                    'spl': metrics.spl,
                    'navigation_collision': metrics.navigation_collision,
                    'collision_rate': metrics.collision_rate,
                    'mean_episode_length': metrics.mean_episode_length,
                    'mean_episode_time': metrics.mean_episode_time,
                    'velocity_smoothness': metrics.velocity_smoothness,
                    'action_smoothness': metrics.action_smoothness,
                    'reward_mean': metrics.reward_mean,
                    'reward_std': metrics.reward_std,
                    'num_episodes': metrics.num_episodes
                }
                for algo_name, metrics in evaluations.items()
            }
        }
        
        filename = f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.file_manager.save_results(report_data, filename)
        self.logger.info(f"Comparison report saved to {filepath}")
    
    def export_evaluation_history(self, filename: Optional[str] = None) -> str:
        """
        导出评估历史
        
        Args:
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"evaluation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'evaluation_history': self.evaluation_history,
            'export_metadata': {
                'export_time': datetime.now().isoformat(),
                'total_evaluations': len(self.evaluation_history),
                'config': self.config
            }
        }
        
        filepath = self.file_manager.save_results(export_data, filename)
        self.logger.info(f"Evaluation history exported to {filepath}")
        
        return filepath
    
    def get_best_performance_summary(self, recent_evaluations: int = 5) -> Dict[str, Any]:
        """
        获取最佳性能摘要
        
        Args:
            recent_evaluations: 考虑最近N次评估
            
        Returns:
            性能摘要字典
        """
        if not self.evaluation_history:
            return {'message': 'No evaluation history available'}
        
        # 获取最近的评估
        recent_evals = self.evaluation_history[-recent_evaluations:]
        
        # 找出各指标的最佳值
        best_success_rate = max(eval_record['metrics'].success_rate 
                               for eval_record in recent_evals)
        best_spl = max(eval_record['metrics'].spl 
                      for eval_record in recent_evals)
        best_navigation_error = min(eval_record['metrics'].navigation_error 
                                   for eval_record in recent_evals)
        
        summary = {
            'recent_evaluations_count': len(recent_evals),
            'best_success_rate': best_success_rate,
            'best_spl': best_spl,
            'best_navigation_error': best_navigation_error,
            'latest_evaluation': {
                'success_rate': recent_evals[-1]['metrics'].success_rate,
                'navigation_error': recent_evals[-1]['metrics'].navigation_error,
                'spl': recent_evals[-1]['metrics'].spl,
                'timestamp': recent_evals[-1]['timestamp']
            }
        }
        
        return summary