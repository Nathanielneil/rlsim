"""
可视化工具 - 提供训练监控、结果分析和轨迹可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

from .logger import get_logger
from .file_manager import FileManager


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化训练可视化器
        
        Args:
            save_dir: 保存目录
        """
        self.logger = get_logger("training_visualizer")
        self.file_manager = FileManager()
        
        if save_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]
            self.save_dir = project_root / "data" / "results" / "plots"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.logger.info(f"Training visualizer initialized, save_dir: {self.save_dir}")
    
    def plot_training_curves(self, 
                           training_data: Dict[str, List[float]], 
                           title: str = "Training Progress",
                           save_name: Optional[str] = None) -> str:
        """
        绘制训练曲线
        
        Args:
            training_data: 训练数据字典
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # 奖励曲线
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            episodes = range(len(rewards))
            axes[0, 0].plot(episodes, rewards, alpha=0.7, linewidth=1)
            axes[0, 0].plot(episodes, self._smooth_curve(rewards, window=50), 
                          color='red', linewidth=2, label='Moving Average (50)')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 成功率曲线
        if 'success_rates' in training_data:
            success_rates = training_data['success_rates']
            eval_episodes = range(0, len(success_rates) * 100, 100)  # 假设每100回合评估一次
            axes[0, 1].plot(eval_episodes, success_rates, 'o-', linewidth=2, markersize=4)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True)
        
        # 回合长度
        if 'episode_lengths' in training_data:
            lengths = training_data['episode_lengths']
            episodes = range(len(lengths))
            axes[1, 0].plot(episodes, lengths, alpha=0.7, linewidth=1)
            axes[1, 0].plot(episodes, self._smooth_curve(lengths, window=50),
                          color='red', linewidth=2, label='Moving Average (50)')
            axes[1, 0].set_title('Episode Length')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 损失曲线（如果有的话）
        if 'policy_loss' in training_data:
            policy_loss = training_data['policy_loss']
            value_loss = training_data.get('value_loss', [])
            
            update_steps = range(len(policy_loss))
            axes[1, 1].plot(update_steps, policy_loss, label='Policy Loss', linewidth=2)
            if value_loss:
                axes[1, 1].plot(update_steps, value_loss, label='Value Loss', linewidth=2)
            
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved to {save_path}")
        return str(save_path)
    
    def plot_reward_components(self, 
                             reward_data: Dict[str, List[float]],
                             title: str = "Reward Components Analysis",
                             save_name: Optional[str] = None) -> str:
        """
        绘制奖励组件分析图
        
        Args:
            reward_data: 奖励组件数据
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        component_names = list(reward_data.keys())
        colors = sns.color_palette("husl", len(component_names))
        
        # 时序图
        ax = axes[0, 0]
        for i, (component, values) in enumerate(reward_data.items()):
            if values:  # 确保有数据
                steps = range(len(values))
                smoothed = self._smooth_curve(values, window=20)
                ax.plot(steps, smoothed, label=component, color=colors[i], linewidth=2)
        
        ax.set_title('Reward Components Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Value')
        ax.legend()
        ax.grid(True)
        
        # 箱线图
        ax = axes[0, 1]
        data_for_box = [values for values in reward_data.values() if values]
        labels_for_box = [name for name, values in reward_data.items() if values]
        
        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        ax.set_title('Reward Components Distribution')
        ax.set_ylabel('Reward Value')
        ax.tick_params(axis='x', rotation=45)
        
        # 相关性热图
        ax = axes[1, 0]
        if len(reward_data) > 1:
            # 创建DataFrame进行相关性分析
            df_data = {}
            min_length = min(len(values) for values in reward_data.values() if values)
            
            for component, values in reward_data.items():
                if values and len(values) >= min_length:
                    df_data[component] = values[:min_length]
            
            if len(df_data) > 1:
                df = pd.DataFrame(df_data)
                correlation_matrix = df.corr()
                
                im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(correlation_matrix.columns)))
                ax.set_yticks(range(len(correlation_matrix.columns)))
                ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                ax.set_yticklabels(correlation_matrix.columns)
                
                # 添加数值标注
                for i in range(len(correlation_matrix)):
                    for j in range(len(correlation_matrix.columns)):
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax)
        
        ax.set_title('Reward Components Correlation')
        
        # 贡献度饼图
        ax = axes[1, 1]
        if reward_data:
            # 计算每个组件的平均绝对贡献
            contributions = {}
            for component, values in reward_data.items():
                if values:
                    contributions[component] = np.mean(np.abs(values))
            
            if contributions:
                labels = list(contributions.keys())
                sizes = list(contributions.values())
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.set_title('Average Absolute Contribution')
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"reward_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Reward components plot saved to {save_path}")
        return str(save_path)
    
    def plot_algorithm_comparison(self, 
                                comparison_data: Dict[str, Dict[str, float]],
                                metrics: List[str],
                                title: str = "Algorithm Comparison",
                                save_name: Optional[str] = None) -> str:
        """
        绘制算法对比图
        
        Args:
            comparison_data: 对比数据 {algorithm: {metric: value}}
            metrics: 要对比的指标列表
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        algorithms = list(comparison_data.keys())
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=(5 * ((num_metrics + 1) // 2), 10))
        fig.suptitle(title, fontsize=16)
        
        # 如果只有一行，确保axes是2D数组
        if num_metrics <= 2:
            axes = axes.reshape(2, -1)
        
        colors = sns.color_palette("husl", len(algorithms))
        
        for i, metric in enumerate(metrics):
            row = i // ((num_metrics + 1) // 2)
            col = i % ((num_metrics + 1) // 2)
            
            if row >= axes.shape[0] or col >= axes.shape[1]:
                continue
                
            ax = axes[row, col]
            
            # 获取数据
            values = [comparison_data[algo].get(metric, 0) for algo in algorithms]
            
            # 条形图
            bars = ax.bar(algorithms, values, color=colors, alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 隐藏未使用的子图
        for i in range(num_metrics, axes.size):
            row = i // axes.shape[1]
            col = i % axes.shape[1]
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Algorithm comparison plot saved to {save_path}")
        return str(save_path)
    
    def _smooth_curve(self, values: List[float], window: int = 10) -> List[float]:
        """平滑曲线"""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return smoothed


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, save_dir: Optional[str] = None):
        """初始化轨迹可视化器"""
        self.logger = get_logger("trajectory_visualizer")
        
        if save_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]
            self.save_dir = project_root / "data" / "results" / "plots"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Trajectory visualizer initialized, save_dir: {self.save_dir}")
    
    def plot_3d_trajectory(self, 
                          trajectory: List[np.ndarray],
                          start_position: np.ndarray,
                          target_position: np.ndarray,
                          obstacles: Optional[List[Dict[str, Any]]] = None,
                          title: str = "3D Trajectory",
                          save_name: Optional[str] = None) -> str:
        """
        绘制3D轨迹
        
        Args:
            trajectory: 轨迹点列表
            start_position: 起始位置
            target_position: 目标位置
            obstacles: 障碍物信息
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 转换轨迹数据
        traj_array = np.array(trajectory)
        x, y, z = traj_array[:, 0], traj_array[:, 1], traj_array[:, 2]
        
        # 绘制轨迹
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.8, label='Trajectory')
        ax.scatter(x[::5], y[::5], z[::5], c=range(0, len(x), 5), 
                  cmap='viridis', s=20, alpha=0.6)
        
        # 起始点和目标点
        ax.scatter(*start_position, color='green', s=100, marker='o', label='Start')
        ax.scatter(*target_position, color='red', s=100, marker='*', label='Target')
        
        # 绘制障碍物
        if obstacles:
            for obs in obstacles:
                self._draw_obstacle_3d(ax, obs)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # 设置相等的坐标轴比例
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 保存图表
        if save_name is None:
            save_name = f"3d_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"3D trajectory plot saved to {save_path}")
        return str(save_path)
    
    def plot_2d_trajectory_comparison(self, 
                                    trajectories: Dict[str, List[np.ndarray]],
                                    start_position: np.ndarray,
                                    target_position: np.ndarray,
                                    height_slice: float = -5.0,
                                    obstacles: Optional[List[Dict[str, Any]]] = None,
                                    title: str = "2D Trajectory Comparison",
                                    save_name: Optional[str] = None) -> str:
        """
        绘制2D轨迹对比图
        
        Args:
            trajectories: 轨迹字典 {name: trajectory}
            start_position: 起始位置
            target_position: 目标位置
            height_slice: 高度切片
            obstacles: 障碍物信息
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        plt.figure(figsize=(12, 10))
        
        colors = sns.color_palette("husl", len(trajectories))
        
        # 绘制轨迹
        for i, (name, trajectory) in enumerate(trajectories.items()):
            traj_array = np.array(trajectory)
            x, y = traj_array[:, 0], traj_array[:, 1]
            
            plt.plot(x, y, color=colors[i], linewidth=2, alpha=0.8, label=name)
            plt.scatter(x[::10], y[::10], color=colors[i], s=20, alpha=0.6)
        
        # 起始点和目标点
        plt.scatter(start_position[0], start_position[1], 
                   color='green', s=150, marker='o', label='Start', zorder=10)
        plt.scatter(target_position[0], target_position[1], 
                   color='red', s=150, marker='*', label='Target', zorder=10)
        
        # 绘制障碍物
        if obstacles:
            for obs in obstacles:
                self._draw_obstacle_2d(plt.gca(), obs, height_slice)
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f"{title} (Height: {-height_slice:.1f}m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 保存图表
        if save_name is None:
            save_name = f"2d_trajectory_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"2D trajectory comparison plot saved to {save_path}")
        return str(save_path)
    
    def plot_trajectory_analysis(self, 
                               trajectory: List[np.ndarray],
                               velocities: Optional[List[np.ndarray]] = None,
                               actions: Optional[List[np.ndarray]] = None,
                               rewards: Optional[List[float]] = None,
                               title: str = "Trajectory Analysis",
                               save_name: Optional[str] = None) -> str:
        """
        绘制轨迹分析图
        
        Args:
            trajectory: 轨迹点列表
            velocities: 速度列表
            actions: 动作列表
            rewards: 奖励列表
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        traj_array = np.array(trajectory)
        steps = range(len(trajectory))
        
        # 位置变化
        ax = axes[0, 0]
        ax.plot(steps, traj_array[:, 0], label='X', linewidth=2)
        ax.plot(steps, traj_array[:, 1], label='Y', linewidth=2)
        ax.plot(steps, traj_array[:, 2], label='Z', linewidth=2)
        ax.set_title('Position Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True)
        
        # 速度分析
        ax = axes[0, 1]
        if velocities:
            vel_array = np.array(velocities)
            vel_magnitude = np.linalg.norm(vel_array, axis=1)
            ax.plot(steps[:len(velocities)], vel_magnitude, linewidth=2, color='red')
            ax.set_title('Speed Over Time')
            ax.set_xlabel('Step')
            ax.set_ylabel('Speed (m/s)')
        else:
            # 从位置计算速度
            if len(trajectory) > 1:
                speeds = []
                for i in range(1, len(trajectory)):
                    speed = np.linalg.norm(trajectory[i] - trajectory[i-1])
                    speeds.append(speed)
                ax.plot(range(1, len(trajectory)), speeds, linewidth=2, color='red')
                ax.set_title('Speed Over Time (Estimated)')
                ax.set_xlabel('Step')
                ax.set_ylabel('Speed (m/step)')
        ax.grid(True)
        
        # 动作分析
        ax = axes[1, 0]
        if actions:
            action_array = np.array(actions)
            if action_array.ndim == 2:
                for i in range(min(4, action_array.shape[1])):
                    ax.plot(steps[:len(actions)], action_array[:, i], 
                           label=f'Action {i}', linewidth=2)
                ax.set_title('Actions Over Time')
                ax.set_xlabel('Step')
                ax.set_ylabel('Action Value')
                ax.legend()
            else:
                ax.plot(steps[:len(actions)], actions, linewidth=2)
                ax.set_title('Action Over Time')
                ax.set_xlabel('Step')
                ax.set_ylabel('Action')
        ax.grid(True)
        
        # 奖励分析
        ax = axes[1, 1]
        if rewards:
            ax.plot(steps[:len(rewards)], rewards, linewidth=2, color='purple')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # 累积奖励
            cumulative_rewards = np.cumsum(rewards)
            ax2 = ax.twinx()
            ax2.plot(steps[:len(rewards)], cumulative_rewards, 
                    linewidth=2, color='orange', alpha=0.7, label='Cumulative')
            
            ax.set_title('Rewards Over Time')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward', color='purple')
            ax2.set_ylabel('Cumulative Reward', color='orange')
            ax2.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_name is None:
            save_name = f"trajectory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Trajectory analysis plot saved to {save_path}")
        return str(save_path)
    
    def _draw_obstacle_3d(self, ax, obstacle: Dict[str, Any]):
        """在3D图中绘制障碍物"""
        center = np.array(obstacle['center'])
        
        if obstacle['type'] == 'box':
            size = np.array(obstacle['size'])
            # 简化为点云表示
            x_range = np.linspace(center[0] - size[0]/2, center[0] + size[0]/2, 5)
            y_range = np.linspace(center[1] - size[1]/2, center[1] + size[1]/2, 5)
            z_range = np.linspace(center[2] - size[2]/2, center[2] + size[2]/2, 5)
            
            xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
            ax.scatter(xx.ravel()[::8], yy.ravel()[::8], zz.ravel()[::8], 
                      c='red', s=10, alpha=0.3)
    
    def _draw_obstacle_2d(self, ax, obstacle: Dict[str, Any], height_slice: float):
        """在2D图中绘制障碍物"""
        center = np.array(obstacle['center'])
        
        # 检查障碍物是否与高度切片相交
        if obstacle['type'] == 'box':
            size = np.array(obstacle['size'])
            z_min = center[2] - size[2]/2
            z_max = center[2] + size[2]/2
            
            if z_min <= height_slice <= z_max:
                # 绘制矩形
                rect = patches.Rectangle(
                    (center[0] - size[0]/2, center[1] - size[1]/2),
                    size[0], size[1],
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
                )
                ax.add_patch(rect)
        
        elif obstacle['type'] == 'cylinder':
            radius = obstacle['size'][0]
            height = obstacle['size'][1]
            z_min = center[2] - height/2
            z_max = center[2] + height/2
            
            if z_min <= height_slice <= z_max:
                # 绘制圆形
                circle = patches.Circle(
                    (center[0], center[1]), radius,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
                )
                ax.add_patch(circle)


class MetricsVisualizer:
    """指标可视化器"""
    
    def __init__(self, save_dir: Optional[str] = None):
        """初始化指标可视化器"""
        self.logger = get_logger("metrics_visualizer")
        
        if save_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]
            self.save_dir = project_root / "data" / "results" / "plots"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_performance_radar(self, 
                             metrics_data: Dict[str, float],
                             title: str = "Performance Radar Chart",
                             save_name: Optional[str] = None) -> str:
        """
        绘制性能雷达图
        
        Args:
            metrics_data: 指标数据
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        # 准备数据
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        # 标准化值到0-1范围
        normalized_values = []
        for metric, value in zip(metrics, values):
            if 'error' in metric.lower() or 'collision' in metric.lower():
                # 对于误差和碰撞，值越小越好
                normalized_values.append(max(0, 1 - value))
            else:
                # 对于其他指标，值越大越好
                normalized_values.append(min(1, max(0, value)))
        
        # 创建雷达图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        values_plot = normalized_values + [normalized_values[0]]  # 闭合图形
        
        # 绘制雷达图
        ax.plot(angles, values_plot, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values_plot, alpha=0.25, color='blue')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_ylim(0, 1)
        
        # 添加网格和标签
        ax.grid(True)
        ax.set_title(title, size=16, y=1.08)
        
        # 保存图表
        if save_name is None:
            save_name = f"performance_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance radar chart saved to {save_path}")
        return str(save_path)
    
    def create_comprehensive_report(self, 
                                  training_data: Dict[str, Any],
                                  evaluation_results: Dict[str, Any],
                                  trajectory_data: Optional[Dict[str, Any]] = None,
                                  save_name: Optional[str] = None) -> str:
        """
        创建综合可视化报告
        
        Args:
            training_data: 训练数据
            evaluation_results: 评估结果
            trajectory_data: 轨迹数据
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        # 创建多页PDF报告或图像集合
        report_dir = self.save_dir / "comprehensive_reports"
        report_dir.mkdir(exist_ok=True)
        
        if save_name is None:
            save_name = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_files = []
        
        # 1. 训练曲线
        training_viz = TrainingVisualizer(report_dir)
        training_file = training_viz.plot_training_curves(training_data, save_name=f"{save_name}_training.png")
        report_files.append(training_file)
        
        # 2. 性能雷达图
        if 'metrics' in evaluation_results:
            radar_file = self.plot_performance_radar(
                evaluation_results['metrics'], 
                save_name=f"{save_name}_performance.png"
            )
            report_files.append(radar_file)
        
        # 3. 轨迹可视化
        if trajectory_data:
            traj_viz = TrajectoryVisualizer(report_dir)
            if 'trajectories' in trajectory_data:
                for i, traj in enumerate(trajectory_data['trajectories'][:3]):  # 只显示前3条轨迹
                    traj_file = traj_viz.plot_3d_trajectory(
                        traj['trajectory'],
                        traj['start_position'],
                        traj['target_position'],
                        save_name=f"{save_name}_trajectory_{i}.png"
                    )
                    report_files.append(traj_file)
        
        self.logger.info(f"Comprehensive report created with {len(report_files)} files in {report_dir}")
        return str(report_dir)