"""
SAC训练脚本 - Soft Actor-Critic训练
"""

import os
import sys
import argparse
import yaml
import torch
import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.airsim_env import AirSimNavigationEnv
from src.agents.sac_agent import SACAgent
from src.utils.logger import setup_logger
from src.utils.visualization import TrainingVisualizer
from src.data.data_collector import DataCollector


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SAC训练脚本')
    
    # 基础参数
    parser.add_argument('--config', type=str, 
                       default='config/sac_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--episodes', type=int, 
                       default=1500,
                       help='训练回合数')
    parser.add_argument('--device', type=str, 
                       default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')
    
    # 实验相关
    parser.add_argument('--experiment-name', type=str,
                       default=None,
                       help='实验名称')
    parser.add_argument('--resume', type=str,
                       default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--save-freq', type=int,
                       default=100,
                       help='模型保存频率')
    
    # 调试和可视化
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    parser.add_argument('--visualize', action='store_true',
                       help='启用训练过程可视化')
    parser.add_argument('--log-interval', type=int,
                       default=10,
                       help='日志输出间隔')
    
    # 环境参数
    parser.add_argument('--max-episode-steps', type=int,
                       default=500,
                       help='每回合最大步数')
    parser.add_argument('--eval-freq', type=int,
                       default=150,
                       help='评估频率')
    parser.add_argument('--eval-episodes', type=int,
                       default=10,
                       help='评估回合数')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_device(device_arg: str) -> str:
    """设置计算设备"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU训练")
        device = 'cpu'
    
    return device


def setup_experiment_dirs(experiment_name: str) -> dict:
    """设置实验目录"""
    if experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"SAC_{timestamp}"
    
    # 创建目录结构
    dirs = {
        'models': Path(f'models/sac/{experiment_name}'),
        'logs': Path(f'data/logs/{experiment_name}'),
        'results': Path(f'data/results/{experiment_name}'),
        'visualizations': Path(f'data/visualizations/{experiment_name}')
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs, experiment_name


def evaluate_agent(agent: SACAgent, env: AirSimNavigationEnv, 
                  num_episodes: int, logger) -> dict:
    """评估智能体性能"""
    logger.info(f"开始评估，共{num_episodes}回合")
    
    eval_rewards = []
    eval_lengths = []
    eval_successes = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # 使用确定性策略
            action = agent.select_action(observation, deterministic=True)
            next_observation, reward, done, truncated, info = env.step(action)
            
            observation = next_observation
            episode_reward += reward
            episode_length += 1
        
        success = info.get('success', False)
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        eval_successes.append(success)
        
        if episode % 5 == 0:
            logger.info(f"评估回合 {episode+1}/{num_episodes} - "
                       f"奖励: {episode_reward:.2f}, "
                       f"长度: {episode_length}, "
                       f"成功: {'是' if success else '否'}")
    
    # 计算评估指标
    eval_metrics = {
        'mean_reward': sum(eval_rewards) / len(eval_rewards),
        'std_reward': torch.tensor(eval_rewards).std().item(),
        'mean_length': sum(eval_lengths) / len(eval_lengths),
        'success_rate': sum(eval_successes) / len(eval_successes),
        'total_episodes': num_episodes
    }
    
    logger.info(f"评估结果 - 平均奖励: {eval_metrics['mean_reward']:.2f}, "
               f"成功率: {eval_metrics['success_rate']:.2%}")
    
    return eval_metrics


def main():
    """主训练流程"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(args.device)
    print(f"使用设备: {device}")
    
    # 设置实验目录
    dirs, experiment_name = setup_experiment_dirs(args.experiment_name)
    print(f"实验名称: {experiment_name}")
    
    # 设置日志
    logger = setup_logger(
        name='sac_training',
        log_file=dirs['logs'] / 'training.log',
        level='DEBUG' if args.debug else 'INFO'
    )
    logger.info(f"开始SAC训练实验: {experiment_name}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"训练设备: {device}")
    
    # 创建环境
    try:
        logger.info("初始化AirSim环境...")
        env = AirSimNavigationEnv(
            config=config.get('env_config', {}),
            max_episode_steps=args.max_episode_steps,
            debug=args.debug
        )
        logger.info("环境初始化成功")
    except Exception as e:
        logger.error(f"环境初始化失败: {e}")
        return
    
    # 创建智能体
    try:
        logger.info("初始化SAC智能体...")
        config['total_timesteps'] = args.episodes * args.max_episode_steps
        agent = SACAgent(
            env=env,
            config=config,
            device=device
        )
        
        # 设置日志间隔
        agent.log_interval = args.log_interval
        
        logger.info("智能体初始化成功")
        logger.info(f"网络架构: {config.get('algorithm_params', {}).get('net_arch', [256, 256])}")
        logger.info(f"缓冲区大小: {agent.buffer_size}")
        logger.info(f"熵系数: {agent.ent_coef}")
        logger.info(f"目标熵: {agent.target_entropy}")
        logger.info(f"学习率: {agent.learning_rate}")
    except Exception as e:
        logger.error(f"智能体初始化失败: {e}")
        return
    
    # 从检查点恢复
    if args.resume:
        try:
            logger.info(f"从检查点恢复: {args.resume}")
            agent.load_model(args.resume)
            logger.info("检查点加载成功")
        except Exception as e:
            logger.error(f"检查点加载失败: {e}")
            return
    
    # 创建可视化器
    if args.visualize:
        visualizer = TrainingVisualizer()
        training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'entropy_coef': [],
            'policy_loss': [],
            'q_loss': []
        }
    
    # 创建数据收集器
    data_collector = DataCollector(str(dirs['results']))
    
    # 训练循环
    logger.info(f"开始训练，共{args.episodes}回合")
    best_success_rate = 0.0
    recent_rewards = []
    recent_successes = []
    
    try:
        for episode in range(1, args.episodes + 1):
            # 训练一个回合
            episode_result = agent._train_episode(episode)
            
            # 记录数据
            episode_reward = episode_result['total_reward']
            episode_length = episode_result['episode_length']
            success = episode_result['success']
            
            recent_rewards.append(episode_reward)
            recent_successes.append(success)
            
            # 保持最近100回合的记录
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
                recent_successes.pop(0)
            
            # 计算移动平均
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            success_rate = sum(recent_successes) / len(recent_successes)
            
            # 更新可视化数据
            if args.visualize:
                training_data['episode_rewards'].append(episode_reward)
                training_data['episode_lengths'].append(episode_length)
                training_data['success_rates'].append(success_rate)
                
                # 获取当前熵系数
                current_ent_coef = torch.exp(agent.log_ent_coef).item()
                training_data['entropy_coef'].append(current_ent_coef)
            
            # 记录训练信息
            if episode % args.log_interval == 0:
                current_ent_coef = torch.exp(agent.log_ent_coef).item()
                logger.info(f"回合 {episode}/{args.episodes} - "
                           f"奖励: {episode_reward:.2f} (平均: {avg_reward:.2f}), "
                           f"长度: {episode_length}, "
                           f"成功率: {success_rate:.2%}, "
                           f"熵系数: {current_ent_coef:.4f}")
            
            # 保存模型
            if episode % args.save_freq == 0:
                model_path = dirs['models'] / f'checkpoint_{episode}.pth'
                agent.save_model(str(model_path), episode)
                logger.info(f"模型已保存: {model_path}")
            
            # 定期评估
            if episode % args.eval_freq == 0:
                eval_metrics = evaluate_agent(
                    agent, env, args.eval_episodes, logger
                )
                
                # 保存最佳模型
                if eval_metrics['success_rate'] > best_success_rate:
                    best_success_rate = eval_metrics['success_rate']
                    best_model_path = dirs['models'] / 'best_model.pth'
                    agent.save_model(str(best_model_path), episode)
                    logger.info(f"新的最佳模型已保存 (成功率: {best_success_rate:.2%})")
                
                # 收集评估数据
                data_collector.add_evaluation_result({
                    'episode': episode,
                    'eval_reward': eval_metrics['mean_reward'],
                    'eval_success_rate': eval_metrics['success_rate'],
                    'eval_length': eval_metrics['mean_length']
                })
            
            # 更新可视化
            if args.visualize and episode % (args.log_interval * 5) == 0:
                try:
                    # 绘制训练曲线
                    fig_path = dirs['visualizations'] / f'training_curves_{episode}.png'
                    visualizer.plot_training_curves(
                        training_data, 
                        title=f"SAC Training Progress (Episode {episode})",
                        save_path=str(fig_path)
                    )
                    
                    # SAC特有的可视化
                    if len(training_data['entropy_coef']) > 10:
                        entropy_fig_path = dirs['visualizations'] / f'entropy_coef_{episode}.png'
                        visualizer.plot_entropy_coefficient(
                            training_data['entropy_coef'],
                            title=f"SAC Entropy Coefficient (Episode {episode})",
                            save_path=str(entropy_fig_path)
                        )
                except Exception as e:
                    logger.warning(f"可视化更新失败: {e}")
    
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise
    finally:
        # 保存最终模型
        final_model_path = dirs['models'] / 'final_model.pth'
        agent.save_model(str(final_model_path), args.episodes)
        logger.info(f"最终模型已保存: {final_model_path}")
        
        # 保存训练数据
        if args.visualize:
            data_path = dirs['results'] / 'training_data.yaml'
            with open(data_path, 'w', encoding='utf-8') as f:
                yaml.dump(training_data, f, default_flow_style=False)
            logger.info(f"训练数据已保存: {data_path}")
        
        # 最终评估
        logger.info("进行最终评估...")
        final_eval = evaluate_agent(agent, env, args.eval_episodes * 2, logger)
        
        # 保存评估结果
        eval_path = dirs['results'] / 'final_evaluation.yaml'
        with open(eval_path, 'w', encoding='utf-8') as f:
            yaml.dump(final_eval, f, default_flow_style=False)
        
        # 关闭环境
        try:
            env.close()
            logger.info("环境已关闭")
        except Exception as e:
            logger.warning(f"环境关闭失败: {e}")
        
        logger.info(f"SAC训练完成! 实验: {experiment_name}")
        logger.info(f"最终成功率: {final_eval['success_rate']:.2%}")
        logger.info(f"最终平均奖励: {final_eval['mean_reward']:.2f}")
        logger.info(f"最终熵系数: {torch.exp(agent.log_ent_coef).item():.4f}")


if __name__ == '__main__':
    main()