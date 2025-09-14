"""
PPO训练脚本 - 训练PPO智能体进行无人机导航
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.environment.airsim_env import AirSimEnv
from src.agents.ppo_agent import PPOAgent
from src.data.data_collector import DataCollector
from src.evaluation.performance_evaluator import PerformanceEvaluator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_training_logger
from src.utils.visualization import TrainingVisualizer
from src.utils.file_manager import FileManager


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train PPO agent for UAV navigation")
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config directory (default: project_root/config)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Training device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--eval-freq', type=int, default=None,
                       help='Evaluation frequency in episodes (overrides config)')
    parser.add_argument('--save-freq', type=int, default=None,
                       help='Model save frequency in episodes (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to model checkpoint to resume training')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging and saving')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering during training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    
    return parser.parse_args()


def setup_environment(config_loader: ConfigLoader):
    """设置训练环境"""
    try:
        # 创建AirSim环境
        env = AirSimEnv(config_path=config_loader.config_dir, algorithm="ppo")
        return env
    except Exception as e:
        raise RuntimeError(f"Failed to setup environment: {e}")


def setup_agent(env, config: dict, device: str = None):
    """设置PPO智能体"""
    try:
        agent = PPOAgent(env, config, device)
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to setup PPO agent: {e}")


def main():
    """主训练函数"""
    args = parse_arguments()
    
    # 设置日志
    experiment_name = args.experiment_name or f"ppo_training"
    logger = setup_training_logger("ppo")
    
    logger.info("="*60)
    logger.info("Starting PPO Training for UAV Navigation")
    logger.info("="*60)
    
    try:
        # 1. 加载配置
        logger.info("Loading configuration...")
        config_loader = ConfigLoader(args.config)
        config = config_loader.load_algorithm_config("ppo")
        
        # 命令行参数覆盖配置
        if args.episodes:
            if 'training' not in config:
                config['training'] = {}
            config['training']['total_episodes'] = args.episodes
            
        if args.eval_freq:
            if 'training' not in config:
                config['training'] = {}
            config['training']['eval_freq'] = args.eval_freq
            
        if args.save_freq:
            if 'training' not in config:
                config['training'] = {}
            config['training']['save_freq'] = args.save_freq
        
        logger.info(f"Configuration loaded successfully")
        if args.debug:
            logger.info(f"Config: {config}")
        
        # 2. 设置环境
        logger.info("Setting up AirSim environment...")
        env = setup_environment(config_loader)
        logger.info(f"Environment created: obs_space={env.observation_space}, "
                   f"action_space={env.action_space}")
        
        # 3. 设置智能体
        logger.info("Setting up PPO agent...")
        agent = setup_agent(env, config, args.device)
        logger.info(f"PPO agent created on device: {agent.device}")
        
        # 4. 加载检查点（如果指定）
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            agent.load_model(args.resume)
            logger.info("Checkpoint loaded successfully")
        
        # 5. 设置数据收集和评估
        logger.info("Setting up data collection and evaluation...")
        data_collector = DataCollector({
            'collect_detailed_data': True,
            'save_trajectories': True
        })
        
        evaluator = PerformanceEvaluator({
            'success_threshold': 3.0,
            'oracle_threshold': 3.0
        })
        
        visualizer = TrainingVisualizer()
        file_manager = FileManager()
        
        # 6. 开始训练
        total_episodes = config.get('training', {}).get('total_episodes', 2000)
        eval_freq = config.get('training', {}).get('eval_freq', 100)
        save_freq = config.get('training', {}).get('save_freq', 500)
        
        logger.info(f"Starting training for {total_episodes} episodes")
        logger.info(f"Evaluation frequency: {eval_freq} episodes")
        logger.info(f"Save frequency: {save_freq} episodes")
        
        # 训练循环
        best_success_rate = 0.0
        training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'evaluation_results': []
        }
        
        for episode in range(total_episodes):
            try:
                # 训练一个回合
                observation, info = env.reset()
                start_position = info.get('current_position', [0, 0, -2])
                target_position = info.get('target_position', [10, 10, -5])
                
                data_collector.start_episode(episode, start_position, target_position)
                
                episode_reward = 0.0
                episode_length = 0
                done = False
                truncated = False
                
                while not (done or truncated):
                    action = agent.select_action(observation, deterministic=False)
                    next_observation, reward, done, truncated, info = env.step(action)
                    
                    # 收集数据
                    data_collector.collect_step(
                        observation=observation,
                        action=action,
                        reward=reward,
                        done=done,
                        truncated=truncated,
                        info=info
                    )
                    
                    observation = next_observation
                    episode_reward += reward
                    episode_length += 1
                
                # 结束回合
                success = info.get('success', False)
                termination_reason = info.get('termination_reason', 'unknown')
                episode_data = data_collector.end_episode(success, termination_reason)
                
                # 记录训练数据
                training_data['episode_rewards'].append(episode_reward)
                training_data['episode_lengths'].append(episode_length)
                
                # 定期日志
                if episode % 10 == 0:
                    recent_rewards = training_data['episode_rewards'][-10:]
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                               f"Length={episode_length}, Success={success}, "
                               f"Avg10={avg_reward:.2f}")
                
                # 定期评估
                if episode > 0 and episode % eval_freq == 0:
                    logger.info(f"Running evaluation at episode {episode}...")
                    eval_metrics = evaluator.evaluate_agent(agent, env, num_episodes=20)
                    
                    success_rate = eval_metrics.success_rate
                    training_data['success_rates'].append(success_rate)
                    training_data['evaluation_results'].append(eval_metrics)
                    
                    logger.info(f"Evaluation - Success Rate: {success_rate:.2%}, "
                               f"Navigation Error: {eval_metrics.navigation_error:.2f}m, "
                               f"SPL: {eval_metrics.spl:.3f}")
                    
                    # 保存最佳模型
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_model_path = file_manager.save_model(
                            agent, "ppo", episode, 
                            metadata={'success_rate': success_rate, 'type': 'best'}
                        )
                        logger.info(f"New best model saved: {best_model_path} "
                                   f"(Success Rate: {success_rate:.2%})")
                
                # 定期保存
                if episode > 0 and episode % save_freq == 0:
                    checkpoint_path = file_manager.save_model(
                        agent, "ppo", episode,
                        metadata={'type': 'checkpoint'}
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                    
                    # 生成可视化
                    if training_data['episode_rewards']:
                        try:
                            plot_path = visualizer.plot_training_curves(
                                training_data, 
                                title=f"PPO Training Progress (Episode {episode})"
                            )
                            logger.info(f"Training curves saved: {plot_path}")
                        except Exception as e:
                            logger.warning(f"Failed to generate training plots: {e}")
            
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in episode {episode}: {e}")
                if args.debug:
                    logger.error(traceback.format_exc())
                continue
        
        # 7. 训练完成后的处理
        logger.info("Training completed!")
        
        # 最终评估
        logger.info("Running final evaluation...")
        final_metrics = evaluator.evaluate_agent(agent, env, num_episodes=100)
        logger.info(f"Final Evaluation Results:")
        logger.info(f"  Success Rate: {final_metrics.success_rate:.2%}")
        logger.info(f"  Oracle Success Rate: {final_metrics.oracle_success_rate:.2%}")
        logger.info(f"  Navigation Error: {final_metrics.navigation_error:.2f}m")
        logger.info(f"  SPL: {final_metrics.spl:.3f}")
        logger.info(f"  Collision Rate: {final_metrics.collision_rate:.2%}")
        
        # 保存最终模型
        final_model_path = file_manager.save_model(
            agent, "ppo", total_episodes,
            metadata={'type': 'final', 'final_metrics': final_metrics}
        )
        logger.info(f"Final model saved: {final_model_path}")
        
        # 导出训练统计
        stats_path = data_collector.export_statistics()
        logger.info(f"Training statistics exported: {stats_path}")
        
        # 生成最终可视化
        try:
            final_plot_path = visualizer.plot_training_curves(
                training_data,
                title="PPO Training - Final Results"
            )
            logger.info(f"Final training curves saved: {final_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate final plots: {e}")
        
        logger.info("="*60)
        logger.info("PPO Training Completed Successfully!")
        logger.info(f"Best Success Rate: {best_success_rate:.2%}")
        logger.info(f"Final Success Rate: {final_metrics.success_rate:.2%}")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
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