"""
日志系统 - 统一的日志记录和管理
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


class Logger:
    """统一日志管理器"""
    
    _instances: Dict[str, 'Logger'] = {}
    
    def __new__(cls, name: str = "dvln_baseline"):
        """单例模式，每个名称只创建一个实例"""
        if name not in cls._instances:
            cls._instances[name] = super(Logger, cls).__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str = "dvln_baseline"):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
        """
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
        
        self._initialized = True
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 创建日志目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]
        log_dir = project_root / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（自动轮转）
        log_file = log_dir / f"{self.name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志单独文件
        error_log_file = log_dir / f"{self.name}_error.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录一般信息"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告信息"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误信息"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误信息"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(message, **kwargs)
    
    def log_training_step(self, step: int, metrics: Dict[str, float]):
        """记录训练步骤信息"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Training Step {step} - {metrics_str}")
    
    def log_episode_end(self, episode: int, total_reward: float, episode_length: int, success: bool):
        """记录回合结束信息"""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Episode {episode} [{status}] - Reward: {total_reward:.2f}, Length: {episode_length}")
    
    def log_model_save(self, model_path: str, episode: int, total_steps: int):
        """记录模型保存信息"""
        self.info(f"Model saved: {model_path} (Episode: {episode}, Steps: {total_steps})")
    
    def log_evaluation(self, success_rate: float, avg_reward: float, avg_length: float):
        """记录评估结果"""
        self.info(f"Evaluation - Success Rate: {success_rate:.2%}, "
                 f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")


# 便捷函数
def get_logger(name: str = "dvln_baseline") -> Logger:
    """获取日志器实例"""
    return Logger(name)


def setup_training_logger(algorithm: str) -> Logger:
    """为特定算法设置训练日志器"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_name = f"{algorithm}_{timestamp}"
    return Logger(logger_name)