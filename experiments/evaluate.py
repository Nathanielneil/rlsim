"""
评估脚本 - 评估训练好的智能体性能
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.environment.airsim_env import AirSimNavigationEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.sac_agent import SACAgent
from src.evaluation.performance_evaluator import PerformanceEvaluator
from src.evaluation.metrics_calculator import MetricsCalculator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.visualization import TrajectoryVisualizer, MetricsVisualizer
from src.utils.file_manager import FileManager


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate trained UAV navigation agent")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       help='Algorithm type: ppo, dqn, sac (default: ppo)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config directory (default: project_root/config)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--device', type=str, default=None,
                       help='Evaluation device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic policy for evaluation')
    parser.add_argument('--save-trajectories', action='store_true',
                       help='Save trajectory data for analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible evaluation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def setup_environment(config: dict, max_episode_steps: int = 500, debug: bool = False):
    """设置评估环境"""
    try:
        env = AirSimNavigationEnv(
            config=config.get('env_config', {}),
            max_episode_steps=max_episode_steps,
            debug=debug
        )
        return env
    except Exception as e:
        raise RuntimeError(f"Failed to setup environment: {e}")


def setup_agent(env, algorithm: str, config: dict, device: str = None):
    """设置智能体"""
    try:
        algorithm_lower = algorithm.lower()
        if algorithm_lower == 'ppo':
            agent = PPOAgent(env, config, device)
        elif algorithm_lower == 'dqn':
            agent = DQNAgent(env, config, device)
        elif algorithm_lower == 'sac':
            agent = SACAgent(env, config, device)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: ppo, dqn, sac")
        
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to setup {algorithm} agent: {e}")


def load_model(agent, model_path: str, logger):
    """加载模型"""
    try:
        agent.load_model(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def run_detailed_evaluation(agent, env, evaluator: PerformanceEvaluator, 
                           num_episodes: int, save_trajectories: bool,
                           logger) -> Dict[str, Any]:
    """运行详细评估"""
    logger.info(f"Starting detailed evaluation for {num_episodes} episodes...")
    
    # 运行评估
    metrics = evaluator.evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
        deterministic=True,
        save_trajectories=save_trajectories
    )
    
    logger.info("Detailed evaluation completed")
    return metrics


def calculate_professional_metrics(trajectories_data: list, 
                                 targets: list,
                                 starts: list,
                                 successes: list,
                                 collisions: list,
                                 logger) -> Dict[str, Any]:
    """计算专业导航指标"""
    logger.info("Calculating professional navigation metrics...")
    
    calculator = MetricsCalculator()
    
    # 计算导航指标
    nav_metrics = calculator.calculate_navigation_metrics(
        trajectories=trajectories_data,
        target_positions=targets,
        start_positions=starts,
        success_flags=successes,
        collision_data=collisions
    )
    
    # 计算效率指标
    episode_times = [1.0] * len(trajectories_data)  # 模拟时间数据
    actions = [[[0, 0, 0, 0]] * len(traj) for traj in trajectories_data]  # 模拟动作数据
    
    eff_metrics = calculator.calculate_efficiency_metrics(
        trajectories=trajectories_data,
        episode_times=episode_times,
        action_data=actions
    )
    
    # 计算安全指标
    scene_bounds = {
        'x_min': -100, 'x_max': 100,
        'y_min': -100, 'y_max': 100,
        'z_min': 2, 'z_max': 15
    }
    
    safety_metrics = calculator.calculate_safety_metrics(
        collision_data=collisions,
        trajectories=trajectories_data,
        scene_bounds=scene_bounds
    )
    
    # 生成报告
    report = calculator.generate_metrics_report(nav_metrics, eff_metrics, safety_metrics)
    
    logger.info("Professional metrics calculation completed")
    return report


def generate_visualizations(evaluation_results: Dict[str, Any],
                          trajectories_data: list,
                          output_dir: Path,
                          logger) -> Dict[str, str]:
    """生成可视化"""
    logger.info("Generating visualizations...")
    
    visualization_files = {}
    
    try:
        # 1. 性能雷达图
        metrics_viz = MetricsVisualizer(output_dir)
        if 'navigation_metrics' in evaluation_results:
            radar_file = metrics_viz.plot_performance_radar(
                evaluation_results['navigation_metrics'],
                title="Agent Performance Evaluation"
            )
            visualization_files['performance_radar'] = radar_file
        
        # 2. 轨迹可视化
        if trajectories_data:
            traj_viz = TrajectoryVisualizer(output_dir)
            
            # 选择几条代表性轨迹进行可视化
            num_trajs_to_plot = min(5, len(trajectories_data))
            for i in range(num_trajs_to_plot):
                traj_data = trajectories_data[i]
                
                # 3D轨迹图
                traj_3d_file = traj_viz.plot_3d_trajectory(
                    trajectory=traj_data['trajectory'],
                    start_position=traj_data['start_position'],
                    target_position=traj_data['target_position'],
                    title=f"3D Trajectory - Episode {i+1}",
                    save_name=f"trajectory_3d_ep_{i+1}.png"
                )
                visualization_files[f'trajectory_3d_{i+1}'] = traj_3d_file
                
                # 轨迹分析图
                if 'rewards' in traj_data:
                    analysis_file = traj_viz.plot_trajectory_analysis(
                        trajectory=traj_data['trajectory'],
                        rewards=traj_data['rewards'],
                        title=f"Trajectory Analysis - Episode {i+1}",
                        save_name=f"trajectory_analysis_ep_{i+1}.png"
                    )
                    visualization_files[f'trajectory_analysis_{i+1}'] = analysis_file
        
        logger.info(f"Generated {len(visualization_files)} visualization files")
        
    except Exception as e:
        logger.warning(f"Error generating visualizations: {e}")
    
    return visualization_files


def save_evaluation_report(evaluation_results: Dict[str, Any],
                          professional_metrics: Dict[str, Any],
                          model_path: str,
                          args,
                          output_file: Path,
                          logger):
    """保存评估报告"""
    logger.info("Saving evaluation report...")
    
    report = {
        'evaluation_info': {
            'model_path': model_path,
            'algorithm': args.algorithm,
            'num_episodes': args.episodes,
            'deterministic': args.deterministic,
            'evaluation_date': str(Path(output_file).stem)
        },
        'basic_metrics': {
            'success_rate': evaluation_results.success_rate,
            'oracle_success_rate': evaluation_results.oracle_success_rate,
            'navigation_error': evaluation_results.navigation_error,
            'trajectory_length': evaluation_results.trajectory_length,
            'spl': evaluation_results.spl,
            'collision_rate': evaluation_results.collision_rate,
            'mean_episode_length': evaluation_results.mean_episode_length,
            'reward_mean': evaluation_results.reward_mean,
            'reward_std': evaluation_results.reward_std
        },
        'professional_metrics': professional_metrics,
        'summary': {
            'overall_performance': _classify_performance(evaluation_results),
            'key_strengths': _identify_strengths(evaluation_results),
            'improvement_areas': _identify_weaknesses(evaluation_results)
        }
    }
    
    file_manager = FileManager()
    report_path = file_manager.save_results(report, output_file.name)
    logger.info(f"Evaluation report saved: {report_path}")
    
    return report_path


def _classify_performance(metrics) -> str:
    """性能分类"""
    success_rate = metrics.success_rate
    spl = metrics.spl
    
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


def _identify_strengths(metrics) -> list:
    """识别优势"""
    strengths = []
    
    if metrics.success_rate >= 0.8:
        strengths.append("High success rate")
    if metrics.spl >= 0.7:
        strengths.append("Efficient path planning")
    if metrics.collision_rate <= 0.1:
        strengths.append("Good collision avoidance")
    if metrics.navigation_error <= 2.0:
        strengths.append("Accurate navigation")
    
    return strengths if strengths else ["Functional basic navigation"]


def _identify_weaknesses(metrics) -> list:
    """识别不足"""
    weaknesses = []
    
    if metrics.success_rate < 0.5:
        weaknesses.append("Low success rate needs improvement")
    if metrics.spl < 0.4:
        weaknesses.append("Inefficient path planning")
    if metrics.collision_rate > 0.3:
        weaknesses.append("High collision rate")
    if metrics.navigation_error > 5.0:
        weaknesses.append("Poor navigation accuracy")
    
    return weaknesses if weaknesses else ["Minor optimization opportunities"]


def main():
    """主评估函数"""
    args = parse_arguments()
    
    # 设置日志
    logger = get_logger("evaluation")
    
    logger.info("="*60)
    logger.info("Starting Agent Evaluation for UAV Navigation")
    logger.info("="*60)
    
    try:
        # 1. 验证模型文件
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Evaluating model: {model_path}")
        logger.info(f"Algorithm: {args.algorithm}")
        logger.info(f"Episodes: {args.episodes}")
        logger.info(f"Deterministic: {args.deterministic}")
        
        # 2. 设置输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = project_root / "data" / "results" / f"evaluation_{model_path.stem}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # 3. 加载配置
        logger.info("Loading configuration...")
        import yaml
        config_path = args.config or f"config/{args.algorithm.lower()}_config.yaml"
        if not os.path.exists(config_path):
            config_path = f"config/{args.algorithm.lower()}_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default config")
            config = {}
        
        # 4. 设置环境
        logger.info("Setting up environment...")
        env = setup_environment(config, max_episode_steps=500, debug=args.verbose)
        
        # 5. 设置智能体
        logger.info("Setting up agent...")
        agent = setup_agent(env, args.algorithm, config, args.device)
        
        # 6. 加载模型
        logger.info("Loading model...")
        load_model(agent, str(model_path), logger)
        
        # 7. 设置评估器
        evaluator = PerformanceEvaluator({
            'success_threshold': 3.0,
            'oracle_threshold': 3.0
        })
        
        # 8. 运行评估
        evaluation_results = run_detailed_evaluation(
            agent, env, evaluator, args.episodes, args.save_trajectories, logger
        )
        
        # 9. 打印基础结果
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Success Rate:           {evaluation_results.success_rate:.2%}")
        logger.info(f"Oracle Success Rate:    {evaluation_results.oracle_success_rate:.2%}")
        logger.info(f"Navigation Error:       {evaluation_results.navigation_error:.2f} m")
        logger.info(f"Trajectory Length:      {evaluation_results.trajectory_length:.2f} m")
        logger.info(f"SPL:                    {evaluation_results.spl:.3f}")
        logger.info(f"Collision Rate:         {evaluation_results.collision_rate:.2%}")
        logger.info(f"Mean Episode Length:    {evaluation_results.mean_episode_length:.1f} steps")
        logger.info(f"Mean Episode Time:      {evaluation_results.mean_episode_time:.1f} s")
        logger.info(f"Mean Reward:            {evaluation_results.reward_mean:.2f}")
        logger.info(f"Reward Std:             {evaluation_results.reward_std:.2f}")
        logger.info("="*50)
        
        # 10. 计算专业指标（如果有轨迹数据）
        professional_metrics = {}
        trajectories_data = []
        
        if args.save_trajectories:
            logger.info("Professional metrics calculation skipped (requires trajectory data)")
            # 这里可以添加更详细的轨迹数据收集和专业指标计算
        
        # 11. 生成可视化
        visualization_files = {}
        if args.visualize:
            visualization_files = generate_visualizations(
                evaluation_results.__dict__, trajectories_data, output_dir, logger
            )
        
        # 12. 保存评估报告
        report_file = output_dir / f"evaluation_report_{model_path.stem}.json"
        report_path = save_evaluation_report(
            evaluation_results.__dict__, professional_metrics,
            str(model_path), args, report_file, logger
        )
        
        # 13. 总结
        performance_level = _classify_performance(evaluation_results)
        logger.info(f"\nOverall Performance Level: {performance_level}")
        
        strengths = _identify_strengths(evaluation_results)
        logger.info(f"Key Strengths: {', '.join(strengths)}")
        
        weaknesses = _identify_weaknesses(evaluation_results)
        logger.info(f"Improvement Areas: {', '.join(weaknesses)}")
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation Completed Successfully!")
        logger.info(f"Results saved to: {output_dir}")
        if visualization_files:
            logger.info(f"Generated {len(visualization_files)} visualization files")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # 清理资源
        try:
            if 'env' in locals():
                env.close()
            if 'agent' in locals():
                agent.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()