"""
指标计算器 - 计算UAV导航任务的专业评估指标
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger


@dataclass
class NavigationMetrics:
    """导航指标数据类"""
    # 成功指标
    success_rate: float              # SR: 成功率
    oracle_success_rate: float       # OSR: Oracle成功率
    
    # 路径指标
    trajectory_length: float         # TL: 轨迹长度
    navigation_error: float          # NE: 导航误差
    spl: float                      # SPL: 路径长度优化的成功率
    
    # 碰撞指标  
    navigation_collision: float      # N-C: 导航碰撞率
    waypoint_collision: float       # W-C: 路径点碰撞率
    dynamic_collision_sr: float     # D-C SR: 动态碰撞成功率
    
    # 四旋翼特定指标
    flight_stability: float         # 飞行稳定性
    altitude_accuracy: float        # 高度控制精度


class MetricsCalculator:
    """指标计算器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化指标计算器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = get_logger("metrics_calculator")
        
        # 阈值设置
        self.success_threshold = self.config.get('success_threshold', 3.0)
        self.oracle_threshold = self.config.get('oracle_threshold', 3.0)
        self.collision_threshold = self.config.get('collision_threshold', 0.5)
        self.stability_window = self.config.get('stability_window', 10)
        
        self.logger.info("Metrics calculator initialized")
    
    def calculate_navigation_metrics(self, 
                                   trajectories: List[List[np.ndarray]],
                                   target_positions: List[np.ndarray],
                                   start_positions: List[np.ndarray],
                                   success_flags: List[bool],
                                   collision_data: List[List[bool]],
                                   velocity_data: Optional[List[List[np.ndarray]]] = None,
                                   action_data: Optional[List[List[np.ndarray]]] = None) -> NavigationMetrics:
        """
        计算导航指标
        
        Args:
            trajectories: 轨迹列表，每个轨迹是位置点列表
            target_positions: 目标位置列表
            start_positions: 起始位置列表
            success_flags: 成功标志列表
            collision_data: 碰撞数据列表
            velocity_data: 速度数据列表（可选）
            action_data: 动作数据列表（可选）
            
        Returns:
            导航指标
        """
        num_episodes = len(trajectories)
        
        if num_episodes == 0:
            self.logger.warning("No trajectories provided for metrics calculation")
            return NavigationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # 1. 成功率 (SR)
        success_rate = np.mean(success_flags)
        
        # 2. Oracle成功率 (OSR)
        oracle_success_rate = self._calculate_oracle_success_rate(
            trajectories, target_positions
        )
        
        # 3. 轨迹长度 (TL)
        trajectory_lengths = [self._calculate_trajectory_length(traj) for traj in trajectories]
        mean_trajectory_length = np.mean(trajectory_lengths)
        
        # 4. 导航误差 (NE)
        navigation_errors = [
            np.linalg.norm(traj[-1] - target) 
            for traj, target in zip(trajectories, target_positions)
        ]
        mean_navigation_error = np.mean(navigation_errors)
        
        # 5. SPL (Success weighted by Path Length)
        spl = self._calculate_spl(
            trajectories, start_positions, target_positions, success_flags
        )
        
        # 6. 导航碰撞 (N-C)
        navigation_collision = self._calculate_navigation_collision(collision_data)
        
        # 7. 路径点碰撞 (W-C)
        waypoint_collision = self._calculate_waypoint_collision(
            trajectories, collision_data
        )
        
        # 8. 动态碰撞成功率 (D-C SR)
        dynamic_collision_sr = self._calculate_dynamic_collision_sr(
            success_flags, collision_data
        )
        
        # 9. 飞行稳定性
        flight_stability = 0.0
        if velocity_data and action_data:
            flight_stability = self._calculate_flight_stability(velocity_data, action_data)
        
        # 10. 高度控制精度
        altitude_accuracy = self._calculate_altitude_accuracy(trajectories, target_positions)
        
        metrics = NavigationMetrics(
            success_rate=success_rate,
            oracle_success_rate=oracle_success_rate,
            trajectory_length=mean_trajectory_length,
            navigation_error=mean_navigation_error,
            spl=spl,
            navigation_collision=navigation_collision,
            waypoint_collision=waypoint_collision,
            dynamic_collision_sr=dynamic_collision_sr,
            flight_stability=flight_stability,
            altitude_accuracy=altitude_accuracy
        )
        
        self.logger.info(f"Calculated navigation metrics for {num_episodes} episodes")
        return metrics
    
    def _calculate_oracle_success_rate(self, 
                                     trajectories: List[List[np.ndarray]], 
                                     target_positions: List[np.ndarray]) -> float:
        """计算Oracle成功率"""
        oracle_successes = 0
        
        for trajectory, target in zip(trajectories, target_positions):
            # 检查轨迹中是否有任何点距离目标小于阈值
            min_distance = min(
                np.linalg.norm(point - target) for point in trajectory
            )
            
            if min_distance <= self.oracle_threshold:
                oracle_successes += 1
        
        return oracle_successes / len(trajectories)
    
    def _calculate_trajectory_length(self, trajectory: List[np.ndarray]) -> float:
        """计算轨迹长度"""
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            segment_length = np.linalg.norm(trajectory[i] - trajectory[i-1])
            total_length += segment_length
        
        return total_length
    
    def _calculate_spl(self, 
                      trajectories: List[List[np.ndarray]],
                      start_positions: List[np.ndarray],
                      target_positions: List[np.ndarray],
                      success_flags: List[bool]) -> float:
        """计算SPL (Success weighted by Path Length)"""
        spl_values = []
        
        for i, (trajectory, start, target, success) in enumerate(
            zip(trajectories, start_positions, target_positions, success_flags)
        ):
            if success:
                # 计算最优路径长度
                optimal_length = np.linalg.norm(target - start)
                
                # 计算实际轨迹长度
                actual_length = self._calculate_trajectory_length(trajectory)
                
                if actual_length > 0:
                    spl_value = optimal_length / max(optimal_length, actual_length)
                else:
                    spl_value = 0.0
            else:
                spl_value = 0.0
            
            spl_values.append(spl_value)
        
        return np.mean(spl_values)
    
    def _calculate_navigation_collision(self, collision_data: List[List[bool]]) -> float:
        """计算导航碰撞率 (N-C)"""
        total_steps = 0
        collision_steps = 0
        
        for episode_collisions in collision_data:
            total_steps += len(episode_collisions)
            collision_steps += sum(episode_collisions)
        
        if total_steps == 0:
            return 0.0
        
        return collision_steps / total_steps
    
    def _calculate_waypoint_collision(self, 
                                    trajectories: List[List[np.ndarray]],
                                    collision_data: List[List[bool]]) -> float:
        """计算路径点碰撞率 (W-C)"""
        total_waypoints = 0
        collision_waypoints = 0
        
        for trajectory, collisions in zip(trajectories, collision_data):
            total_waypoints += len(trajectory)
            collision_waypoints += sum(collisions[:len(trajectory)])
        
        if total_waypoints == 0:
            return 0.0
        
        return collision_waypoints / total_waypoints
    
    def _calculate_dynamic_collision_sr(self, 
                                      success_flags: List[bool],
                                      collision_data: List[List[bool]]) -> float:
        """计算动态碰撞成功率 (D-C SR)"""
        # 筛选出有碰撞的回合
        collision_episodes = []
        for i, collisions in enumerate(collision_data):
            if any(collisions):  # 如果该回合有碰撞
                collision_episodes.append(success_flags[i])
        
        if not collision_episodes:
            return 1.0  # 没有碰撞回合，成功率为100%
        
        return np.mean(collision_episodes)
    
    def _calculate_flight_stability(self, 
                                  velocity_data: List[List[np.ndarray]],
                                  action_data: List[List[np.ndarray]]) -> float:
        """计算飞行稳定性"""
        velocity_stabilities = []
        action_stabilities = []
        
        for velocities, actions in zip(velocity_data, action_data):
            # 速度稳定性
            if len(velocities) > self.stability_window:
                vel_stability = self._calculate_sequence_stability(velocities)
                velocity_stabilities.append(vel_stability)
            
            # 动作稳定性
            if len(actions) > self.stability_window:
                action_stability = self._calculate_sequence_stability(actions)
                action_stabilities.append(action_stability)
        
        # 综合稳定性
        overall_stability = 0.0
        if velocity_stabilities and action_stabilities:
            overall_stability = (np.mean(velocity_stabilities) + np.mean(action_stabilities)) / 2.0
        elif velocity_stabilities:
            overall_stability = np.mean(velocity_stabilities)
        elif action_stabilities:
            overall_stability = np.mean(action_stabilities)
        
        return overall_stability
    
    def _calculate_sequence_stability(self, sequence: List[np.ndarray]) -> float:
        """计算序列的稳定性"""
        if len(sequence) < 2:
            return 0.0
        
        # 计算变化率
        changes = []
        for i in range(1, len(sequence)):
            change = np.linalg.norm(sequence[i] - sequence[i-1])
            changes.append(change)
        
        if not changes:
            return 1.0
        
        # 稳定性 = 1 / (1 + std(changes))
        stability = 1.0 / (1.0 + np.std(changes))
        return stability
    
    def _calculate_altitude_accuracy(self, 
                                   trajectories: List[List[np.ndarray]],
                                   target_positions: List[np.ndarray]) -> float:
        """计算高度控制精度"""
        altitude_errors = []
        
        for trajectory, target in zip(trajectories, target_positions):
            target_altitude = -target[2]  # AirSim使用负Z作为高度
            
            for point in trajectory:
                current_altitude = -point[2]
                altitude_error = abs(current_altitude - target_altitude)
                altitude_errors.append(altitude_error)
        
        if not altitude_errors:
            return 0.0
        
        # 计算平均高度误差
        mean_altitude_error = np.mean(altitude_errors)
        
        # 转换为精度分数（误差越小精度越高）
        accuracy = 1.0 / (1.0 + mean_altitude_error)
        return accuracy
    
    def calculate_efficiency_metrics(self, 
                                   trajectories: List[List[np.ndarray]],
                                   episode_times: List[float],
                                   action_data: List[List[np.ndarray]]) -> Dict[str, float]:
        """
        计算效率指标
        
        Args:
            trajectories: 轨迹列表
            episode_times: 回合时间列表
            action_data: 动作数据列表
            
        Returns:
            效率指标字典
        """
        metrics = {}
        
        # 平均速度
        avg_speeds = []
        for trajectory, time in zip(trajectories, episode_times):
            if time > 0 and len(trajectory) > 1:
                distance = self._calculate_trajectory_length(trajectory)
                avg_speed = distance / time
                avg_speeds.append(avg_speed)
        
        metrics['average_speed'] = np.mean(avg_speeds) if avg_speeds else 0.0
        
        # 动作效率（动作幅度的一致性）
        action_efficiencies = []
        for actions in action_data:
            if len(actions) > 1:
                action_magnitudes = [np.linalg.norm(action) for action in actions]
                efficiency = 1.0 / (1.0 + np.std(action_magnitudes))
                action_efficiencies.append(efficiency)
        
        metrics['action_efficiency'] = np.mean(action_efficiencies) if action_efficiencies else 0.0
        
        # 路径效率（直线度）
        path_efficiencies = []
        for trajectory in trajectories:
            if len(trajectory) > 2:
                # 计算直线距离与实际路径的比率
                straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
                actual_distance = self._calculate_trajectory_length(trajectory)
                
                if actual_distance > 0:
                    efficiency = straight_distance / actual_distance
                    path_efficiencies.append(efficiency)
        
        metrics['path_efficiency'] = np.mean(path_efficiencies) if path_efficiencies else 0.0
        
        return metrics
    
    def calculate_safety_metrics(self, 
                               collision_data: List[List[bool]],
                               trajectories: List[List[np.ndarray]],
                               scene_bounds: Dict[str, float]) -> Dict[str, float]:
        """
        计算安全指标
        
        Args:
            collision_data: 碰撞数据列表
            trajectories: 轨迹列表
            scene_bounds: 场景边界
            
        Returns:
            安全指标字典
        """
        metrics = {}
        
        # 碰撞频率
        collision_rates = []
        for collisions in collision_data:
            if collisions:
                collision_rate = sum(collisions) / len(collisions)
                collision_rates.append(collision_rate)
        
        metrics['collision_frequency'] = np.mean(collision_rates) if collision_rates else 0.0
        
        # 边界违反
        boundary_violations = 0
        total_points = 0
        
        for trajectory in trajectories:
            for point in trajectory:
                total_points += 1
                
                if (point[0] < scene_bounds['x_min'] or point[0] > scene_bounds['x_max'] or
                    point[1] < scene_bounds['y_min'] or point[1] > scene_bounds['y_max'] or
                    point[2] < scene_bounds['z_min'] or point[2] > scene_bounds['z_max']):
                    boundary_violations += 1
        
        metrics['boundary_violation_rate'] = boundary_violations / total_points if total_points > 0 else 0.0
        
        # 安全边距维持
        safe_distances = []
        for trajectory in trajectories:
            for point in trajectory:
                # 计算到边界的最小距离
                distances = [
                    point[0] - scene_bounds['x_min'],
                    scene_bounds['x_max'] - point[0],
                    point[1] - scene_bounds['y_min'],
                    scene_bounds['y_max'] - point[1],
                    point[2] - scene_bounds['z_min'],
                    scene_bounds['z_max'] - point[2]
                ]
                min_distance = min(distances)
                safe_distances.append(max(0, min_distance))
        
        metrics['mean_safety_margin'] = np.mean(safe_distances) if safe_distances else 0.0
        
        return metrics
    
    def generate_metrics_report(self, 
                               navigation_metrics: NavigationMetrics,
                               efficiency_metrics: Dict[str, float],
                               safety_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        生成完整的指标报告
        
        Args:
            navigation_metrics: 导航指标
            efficiency_metrics: 效率指标
            safety_metrics: 安全指标
            
        Returns:
            指标报告字典
        """
        report = {
            'navigation_metrics': {
                'success_rate': navigation_metrics.success_rate,
                'oracle_success_rate': navigation_metrics.oracle_success_rate,
                'trajectory_length': navigation_metrics.trajectory_length,
                'navigation_error': navigation_metrics.navigation_error,
                'spl': navigation_metrics.spl,
                'navigation_collision': navigation_metrics.navigation_collision,
                'waypoint_collision': navigation_metrics.waypoint_collision,
                'dynamic_collision_sr': navigation_metrics.dynamic_collision_sr,
                'flight_stability': navigation_metrics.flight_stability,
                'altitude_accuracy': navigation_metrics.altitude_accuracy
            },
            'efficiency_metrics': efficiency_metrics,
            'safety_metrics': safety_metrics,
            'summary': {
                'overall_score': self._calculate_overall_score(
                    navigation_metrics, efficiency_metrics, safety_metrics
                ),
                'performance_level': self._classify_performance(navigation_metrics)
            },
            'timestamp': np.datetime64('now').isoformat()
        }
        
        return report
    
    def _calculate_overall_score(self, 
                               nav_metrics: NavigationMetrics,
                               eff_metrics: Dict[str, float],
                               safety_metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        # 权重设置
        nav_weight = 0.6
        eff_weight = 0.2
        safety_weight = 0.2
        
        # 导航得分
        nav_score = (nav_metrics.success_rate * 0.3 +
                    nav_metrics.oracle_success_rate * 0.2 +
                    nav_metrics.spl * 0.3 +
                    (1 - min(nav_metrics.navigation_error / 10.0, 1.0)) * 0.2)
        
        # 效率得分
        eff_score = np.mean(list(eff_metrics.values())) if eff_metrics else 0.5
        
        # 安全得分
        safety_score = (1 - safety_metrics.get('collision_frequency', 0.0)) * 0.5 + \
                      (1 - safety_metrics.get('boundary_violation_rate', 0.0)) * 0.5
        
        overall_score = (nav_score * nav_weight + 
                        eff_score * eff_weight + 
                        safety_score * safety_weight)
        
        return min(1.0, max(0.0, overall_score))
    
    def _classify_performance(self, nav_metrics: NavigationMetrics) -> str:
        """分类性能水平"""
        success_rate = nav_metrics.success_rate
        spl = nav_metrics.spl
        
        if success_rate >= 0.9 and spl >= 0.8:
            return "Excellent"
        elif success_rate >= 0.7 and spl >= 0.6:
            return "Good"
        elif success_rate >= 0.5 and spl >= 0.4:
            return "Fair"
        elif success_rate >= 0.3:
            return "Poor"
        else:
            return "Very Poor"