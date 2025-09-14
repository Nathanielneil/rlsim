"""
算法对比脚本 - 对比多个算法的性能
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.environment.airsim_env import AirSimNavigationEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.sac_agent import SACAgent
from src.evaluation.performance_evaluator import PerformanceEvaluator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.visualization import TrainingVisualizer, MetricsVisualizer
from src.utils.file_manager import FileManager


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Compare multiple UAV navigation algorithms")
    
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--names', type=str, nargs='+',
                       help='Names for the models (default: use filenames)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       help='Algorithm types for each model (default: all ppo)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config directory (default: project_root/config)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes per algorithm (default: 100)')
    parser.add_argument('--device', type=str, default=None,
                       help='Evaluation device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for comparison results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate comparison visualizations')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed analysis with professional metrics')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible comparison')
    
    return parser.parse_args()


def validate_arguments(args):
    """验证命令行参数"""
    # 检查模型文件是否存在
    for model_path in args.models:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 设置默认名称
    if args.names is None:
        args.names = [Path(model).stem for model in args.models]
    elif len(args.names) != len(args.models):
        raise ValueError("Number of names must match number of models")
    
    # 设置默认算法
    if args.algorithms is None:
        args.algorithms = ['ppo'] * len(args.models)
    elif len(args.algorithms) != len(args.models):
        raise ValueError("Number of algorithms must match number of models")
    
    return args


def setup_agent_for_algorithm(env, algorithm: str, config: dict, device: str = None):
    """为特定算法设置智能体"""
    algorithm = algorithm.lower()
    
    if algorithm == 'ppo':
        return PPOAgent(env, config, device)
    elif algorithm == 'dqn':
        return DQNAgent(env, config, device)
    elif algorithm == 'sac':
        return SACAgent(env, config, device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: ppo, dqn, sac")


def evaluate_single_model(model_path: str, 
                         model_name: str,
                         algorithm: str,
                         env,
                         evaluator: PerformanceEvaluator,
                         config: dict,
                         num_episodes: int,
                         device: str,
                         logger) -> Dict[str, Any]:
    """评估单个模型"""
    logger.info(f"Evaluating {model_name} ({algorithm})...")
    
    try:
        # 设置智能体
        agent = setup_agent_for_algorithm(env, algorithm, config, device)
        
        # 加载模型
        agent.load_model(model_path)
        logger.info(f"Model loaded: {model_path}")
        
        # 运行评估
        metrics = evaluator.evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes,
            deterministic=True,
            save_trajectories=False
        )
        
        # 清理智能体
        agent.cleanup()
        
        logger.info(f"{model_name} evaluation completed - "
                   f"Success Rate: {metrics.success_rate:.2%}, "
                   f"SPL: {metrics.spl:.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to evaluate {model_name}: {e}")
        raise


def run_comparison(model_configs: List[Dict[str, str]], 
                  args,
                  logger) -> Dict[str, Any]:
    """运行算法对比"""
    logger.info(f"Starting comparison of {len(model_configs)} models...")
    
    # 设置评估器
    evaluator = PerformanceEvaluator({
        'success_threshold': 3.0,
        'oracle_threshold': 3.0
    })
    
    # 存储所有评估结果
    all_results = {}
    
    for model_config in model_configs:
        model_name = model_config['name']
        model_path = model_config['path']
        algorithm = model_config['algorithm']
        
        logger.info(f"Evaluating {model_name} ({algorithm})...")
        
        try:
            # 加载对应算法的配置
            import yaml
            config_path = f"config/{algorithm}_config.yaml"
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                logger.warning(f"Config file {config_path} not found, using default config")
                config = {}
            
            # 为每个模型创建新环境（确保环境状态独立）
            env = AirSimNavigationEnv(
                config=config.get('env_config', {}),
                max_episode_steps=500,
                debug=False
            )
            
            try:
                # 评估模型
                metrics = evaluate_single_model(
                    model_path=model_path,
                    model_name=model_name,
                    algorithm=algorithm,
                    env=env,
                    evaluator=evaluator,
                    config=config,
                    num_episodes=args.episodes,
                    device=args.device,
                    logger=logger
                )
                
                all_results[model_name] = metrics
                
            finally:
                env.close()
                
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    logger.info("All model evaluations completed")
    return all_results


def generate_comparison_report(results: Dict[str, Any], 
                             model_configs: List[Dict[str, str]],
                             args) -> Dict[str, Any]:
    """生成对比报告"""
    # 提取关键指标进行对比
    comparison_data = {}
    
    for model_name, metrics in results.items():
        comparison_data[model_name] = {
            'success_rate': metrics.success_rate,
            'oracle_success_rate': metrics.oracle_success_rate,
            'navigation_error': metrics.navigation_error,
            'trajectory_length': metrics.trajectory_length,
            'spl': metrics.spl,
            'collision_rate': metrics.collision_rate,
            'mean_episode_length': metrics.mean_episode_length,
            'mean_episode_time': metrics.mean_episode_time,
            'velocity_smoothness': metrics.velocity_smoothness,
            'action_smoothness': metrics.action_smoothness,
            'reward_mean': metrics.reward_mean
        }
    
    # 使用性能评估器的比较功能
    evaluator = PerformanceEvaluator()
    comparison_result = evaluator.compare_algorithms(results, save_report=False)
    
    # 添加额外信息
    comparison_result['model_info'] = {
        config['name']: {
            'algorithm': config['algorithm'],
            'model_path': config['path']
        }
        for config in model_configs
    }
    
    comparison_result['evaluation_settings'] = {
        'episodes_per_model': args.episodes,
        'deterministic_policy': True,
        'random_seed': args.seed
    }
    
    return comparison_result


def print_comparison_results(comparison_result: Dict[str, Any], logger):
    """打印对比结果"""
    logger.info("\n" + "="*80)
    logger.info("ALGORITHM COMPARISON RESULTS")
    logger.info("="*80)
    
    # 总体排名
    if 'overall_ranking' in comparison_result:
        logger.info("\nOverall Ranking:")
        for i, result in enumerate(comparison_result['overall_ranking']):
            logger.info(f"  {i+1}. {result['algorithm']} (Score: {result['score']:.3f})")
    
    # 各项指标的最佳算法
    logger.info("\nBest Performance by Metric:")
    key_metrics = [
        'success_rate', 'oracle_success_rate', 'spl', 
        'navigation_error', 'collision_rate', 'trajectory_length'
    ]
    
    for metric in key_metrics:
        if metric in comparison_result:
            best_algo = comparison_result[metric].get('best_algorithm', 'N/A')
            best_value = comparison_result[metric].get('best_value', 0)
            
            if 'error' in metric or 'collision' in metric:
                logger.info(f"  {metric.replace('_', ' ').title()}: {best_algo} ({best_value:.4f})")
            else:
                logger.info(f"  {metric.replace('_', ' ').title()}: {best_algo} ({best_value:.4f})")
    
    # 详细对比表
    logger.info(f"\nDetailed Comparison:")
    logger.info("-" * 80)
    
    # 表头
    algorithms = list(comparison_result['overall_ranking'][0].keys())
    if 'algorithm' in algorithms:
        algorithms = [r['algorithm'] for r in comparison_result['overall_ranking']]
    
    header = f"{'Metric':<25}"
    for algo in algorithms[:4]:  # 限制显示前4个算法
        header += f"{algo:<15}"
    logger.info(header)
    logger.info("-" * 80)
    
    # 数据行
    display_metrics = [
        ('Success Rate', 'success_rate', '{:.2%}'),
        ('Oracle SR', 'oracle_success_rate', '{:.2%}'),
        ('Nav Error (m)', 'navigation_error', '{:.2f}'),
        ('SPL', 'spl', '{:.3f}'),
        ('Collision Rate', 'collision_rate', '{:.2%}'),
        ('Traj Length (m)', 'trajectory_length', '{:.1f}'),
        ('Reward Mean', 'reward_mean', '{:.2f}')
    ]
    
    for display_name, metric_key, format_str in display_metrics:
        if metric_key in comparison_result:
            row = f"{display_name:<25}"
            for algo in algorithms[:4]:
                if algo in comparison_result[metric_key]:
                    value = comparison_result[metric_key][algo]
                    row += format_str.format(value).rjust(15)
                else:
                    row += "N/A".rjust(15)
            logger.info(row)
    
    logger.info("="*80)


def save_comparison_results(comparison_result: Dict[str, Any],
                           output_dir: Path,
                           args,
                           logger) -> str:
    """保存对比结果"""
    logger.info("Saving comparison results...")
    
    # 保存完整结果
    file_manager = FileManager()
    
    # 添加运行信息
    comparison_result['run_info'] = {
        'models_compared': len(args.models),
        'episodes_per_model': args.episodes,
        'model_paths': args.models,
        'model_names': args.names,
        'algorithms': args.algorithms
    }
    
    # 保存JSON报告
    report_file = f"algorithm_comparison_{len(args.models)}models.json"
    report_path = file_manager.save_results(comparison_result, report_file)
    
    logger.info(f"Comparison results saved: {report_path}")
    return report_path


def generate_comparison_visualizations(comparison_result: Dict[str, Any],
                                     output_dir: Path,
                                     logger) -> List[str]:
    """生成对比可视化"""
    logger.info("Generating comparison visualizations...")
    
    visualization_files = []
    
    try:
        # 1. 算法对比柱状图
        training_viz = TrainingVisualizer(output_dir)
        
        # 准备对比数据
        algorithms = [r['algorithm'] for r in comparison_result['overall_ranking']]
        
        comparison_data = {}
        for algo in algorithms:
            comparison_data[algo] = {}
            for metric in ['success_rate', 'spl', 'navigation_error', 'collision_rate']:
                if metric in comparison_result and algo in comparison_result[metric]:
                    comparison_data[algo][metric] = comparison_result[metric][algo]
        
        if comparison_data:
            metrics_to_plot = ['success_rate', 'spl', 'navigation_error', 'collision_rate']
            comparison_file = training_viz.plot_algorithm_comparison(
                comparison_data,
                metrics_to_plot,
                title="Algorithm Performance Comparison"
            )
            visualization_files.append(comparison_file)
        
        # 2. 性能雷达图（为每个算法）
        metrics_viz = MetricsVisualizer(output_dir)
        
        for algo in algorithms[:3]:  # 最多显示3个算法的雷达图
            algo_metrics = {}
            for metric in ['success_rate', 'oracle_success_rate', 'spl', 
                          'velocity_smoothness', 'action_smoothness']:
                if metric in comparison_result and algo in comparison_result[metric]:
                    algo_metrics[metric] = comparison_result[metric][algo]
            
            if algo_metrics:
                radar_file = metrics_viz.plot_performance_radar(
                    algo_metrics,
                    title=f"{algo} Performance Radar",
                    save_name=f"radar_{algo.lower()}.png"
                )
                visualization_files.append(radar_file)
        
        logger.info(f"Generated {len(visualization_files)} visualization files")
        
    except Exception as e:
        logger.warning(f"Error generating visualizations: {e}")
    
    return visualization_files


def main():
    """主对比函数"""
    args = parse_arguments()
    args = validate_arguments(args)
    
    # 设置日志
    logger = get_logger("algorithm_comparison")
    
    logger.info("="*80)
    logger.info("Starting Algorithm Comparison for UAV Navigation")
    logger.info("="*80)
    
    try:
        # 1. 设置输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = project_root / "data" / "results" / "algorithm_comparison"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # 2. 准备模型配置
        model_configs = []
        for i, (model_path, model_name, algorithm) in enumerate(zip(args.models, args.names, args.algorithms)):
            model_configs.append({
                'path': model_path,
                'name': model_name,
                'algorithm': algorithm.lower()
            })
        
        logger.info(f"Comparing {len(model_configs)} models:")
        for config in model_configs:
            logger.info(f"  - {config['name']} ({config['algorithm']}): {config['path']}")
        
        # 3. 运行对比评估
        logger.info(f"Running evaluation with {args.episodes} episodes per model...")
        results = run_comparison(model_configs, args, logger)
        
        # 5. 生成对比报告
        logger.info("Generating comparison report...")
        comparison_result = generate_comparison_report(results, model_configs, args)
        
        # 6. 打印结果
        print_comparison_results(comparison_result, logger)
        
        # 7. 保存结果
        report_path = save_comparison_results(comparison_result, output_dir, args, logger)
        
        # 8. 生成可视化
        visualization_files = []
        if args.visualize:
            visualization_files = generate_comparison_visualizations(
                comparison_result, output_dir, logger
            )
        
        # 9. 总结
        best_algorithm = comparison_result['overall_ranking'][0]['algorithm']
        best_score = comparison_result['overall_ranking'][0]['score']
        
        logger.info(f"\n" + "="*80)
        logger.info("COMPARISON COMPLETED SUCCESSFULLY!")
        logger.info(f"Best Overall Algorithm: {best_algorithm} (Score: {best_score:.3f})")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Detailed report: {report_path}")
        
        if visualization_files:
            logger.info(f"Generated {len(visualization_files)} visualization files")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Algorithm comparison failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()