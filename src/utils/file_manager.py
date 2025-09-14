"""
文件管理器 - 模型文件和数据文件的统一管理
"""

import os
import shutil
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from .logger import get_logger


class FileManager:
    """文件管理器类"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        初始化文件管理器
        
        Args:
            project_root: 项目根目录路径
        """
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parents[2]
        else:
            self.project_root = Path(project_root)
        
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.logger = get_logger("file_manager")
        
        # 创建必要目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        dirs_to_create = [
            self.models_dir / "ppo",
            self.models_dir / "dqn", 
            self.models_dir / "sac",
            self.data_dir / "trajectories",
            self.data_dir / "logs",
            self.data_dir / "results"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_model_filename(self, algorithm: str, episode: int, timestamp: Optional[str] = None) -> str:
        """
        生成模型文件名
        
        Args:
            algorithm: 算法名称
            episode: 回合数
            timestamp: 时间戳，默认使用当前时间
            
        Returns:
            模型文件名
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{timestamp}_{algorithm}_{episode}.pth"
    
    def save_model(self, model: Any, algorithm: str, episode: int, 
                   metadata: Optional[Dict[str, Any]] = None,
                   timestamp: Optional[str] = None) -> str:
        """
        保存模型文件
        
        Args:
            model: 要保存的模型
            algorithm: 算法名称
            episode: 回合数
            metadata: 元数据
            timestamp: 时间戳
            
        Returns:
            保存的文件路径
        """
        filename = self.generate_model_filename(algorithm, episode, timestamp)
        model_dir = self.models_dir / algorithm.lower()
        model_path = model_dir / filename
        
        # 准备保存的数据
        save_data = {
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
            'algorithm': algorithm,
            'episode': episode,
            'timestamp': timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
            'metadata': metadata or {}
        }
        
        try:
            torch.save(save_data, model_path)
            self.logger.info(f"Model saved: {model_path}")
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        加载模型文件
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型数据
        """
        try:
            model_data = torch.load(model_path, map_location='cpu')
            self.logger.info(f"Model loaded: {model_path}")
            return model_data
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def list_models(self, algorithm: str) -> List[str]:
        """
        列出指定算法的所有模型文件
        
        Args:
            algorithm: 算法名称
            
        Returns:
            模型文件路径列表
        """
        model_dir = self.models_dir / algorithm.lower()
        if not model_dir.exists():
            return []
        
        model_files = list(model_dir.glob("*.pth"))
        return sorted([str(f) for f in model_files])
    
    def get_latest_model(self, algorithm: str) -> Optional[str]:
        """
        获取指定算法的最新模型文件
        
        Args:
            algorithm: 算法名称
            
        Returns:
            最新模型文件路径，如果不存在则返回None
        """
        models = self.list_models(algorithm)
        if not models:
            return None
        
        # 按文件修改时间排序
        models_with_time = [(f, os.path.getmtime(f)) for f in models]
        models_with_time.sort(key=lambda x: x[1], reverse=True)
        
        return models_with_time[0][0]
    
    def cleanup_old_models(self, algorithm: str, keep_count: int = 5):
        """
        清理旧的模型文件，只保留最新的几个
        
        Args:
            algorithm: 算法名称
            keep_count: 保留的文件数量
        """
        models = self.list_models(algorithm)
        if len(models) <= keep_count:
            return
        
        # 按文件修改时间排序
        models_with_time = [(f, os.path.getmtime(f)) for f in models]
        models_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # 删除多余的文件
        for model_path, _ in models_with_time[keep_count:]:
            try:
                os.remove(model_path)
                self.logger.info(f"Deleted old model: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete {model_path}: {e}")
    
    def save_trajectory(self, trajectory_data: Dict[str, Any], filename: str) -> str:
        """
        保存轨迹数据
        
        Args:
            trajectory_data: 轨迹数据
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        trajectory_path = self.data_dir / "trajectories" / filename
        
        try:
            with open(trajectory_path, 'wb') as f:
                pickle.dump(trajectory_data, f)
            self.logger.debug(f"Trajectory saved: {trajectory_path}")
            return str(trajectory_path)
        except Exception as e:
            self.logger.error(f"Failed to save trajectory: {e}")
            raise
    
    def load_trajectory(self, filename: str) -> Dict[str, Any]:
        """
        加载轨迹数据
        
        Args:
            filename: 文件名
            
        Returns:
            轨迹数据
        """
        trajectory_path = self.data_dir / "trajectories" / filename
        
        try:
            with open(trajectory_path, 'rb') as f:
                trajectory_data = pickle.load(f)
            self.logger.debug(f"Trajectory loaded: {trajectory_path}")
            return trajectory_data
        except Exception as e:
            self.logger.error(f"Failed to load trajectory {filename}: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        保存实验结果
        
        Args:
            results: 实验结果数据
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        results_path = self.data_dir / "results" / filename
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Results saved: {results_path}")
            return str(results_path)
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        加载实验结果
        
        Args:
            filename: 文件名
            
        Returns:
            实验结果数据
        """
        results_path = self.data_dir / "results" / filename
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            self.logger.info(f"Results loaded: {results_path}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load results {filename}: {e}")
            raise