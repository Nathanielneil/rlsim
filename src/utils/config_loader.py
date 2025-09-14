"""
配置加载器 - 统一管理所有配置文件的加载
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir: str = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config文件夹
        """
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]  # 上两级目录
            self.config_dir = project_root / "config"
        else:
            self.config_dir = Path(config_dir)
            
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        加载JSON配置文件
        
        Args:
            filename: JSON文件名
            
        Returns:
            配置字典
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            filename: YAML文件名
            
        Returns:
            配置字典
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
    
    def load_airsim_settings(self) -> Dict[str, Any]:
        """加载AirSim设置文件"""
        return self.load_json("settings.json")
    
    def load_scene_config(self) -> Dict[str, Any]:
        """加载场景配置文件"""
        return self.load_yaml("scene_config.yaml")
    
    def load_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """
        加载算法配置文件
        
        Args:
            algorithm: 算法名称 (ppo, dqn, sac)
            
        Returns:
            算法配置字典
        """
        filename = f"{algorithm.lower()}_config.yaml"
        return self.load_yaml(filename)
    
    def save_json(self, data: Dict[str, Any], filename: str) -> None:
        """
        保存数据为JSON文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
        """
        file_path = self.config_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_yaml(self, data: Dict[str, Any], filename: str) -> None:
        """
        保存数据为YAML文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
        """
        file_path = self.config_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置文件，override_config会覆盖base_config中的相同键
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged