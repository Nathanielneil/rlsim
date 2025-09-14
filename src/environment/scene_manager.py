"""
场景管理器 - 管理障碍物、边界和科学的目标点生成
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import random

from ..utils.logger import get_logger


@dataclass
class Obstacle:
    """障碍物数据类"""
    name: str
    type: str  # 'box', 'cylinder', 'sphere'
    center: np.ndarray
    size: np.ndarray  # 对于box: [width, height, depth], cylinder: [radius, height], sphere: [radius]
    

class SceneManager:
    """场景管理器类"""
    
    def __init__(self, scene_config: Dict[str, Any]):
        """
        初始化场景管理器
        
        Args:
            scene_config: 场景配置
        """
        self.logger = get_logger("scene_manager")
        self.scene_config = scene_config
        
        # 场景边界
        self.bounds = scene_config['scene_bounds']
        self.x_min, self.x_max = self.bounds['x_min'], self.bounds['x_max']
        self.y_min, self.y_max = self.bounds['y_min'], self.bounds['y_max']
        self.z_min, self.z_max = self.bounds['z_min'], self.bounds['z_max']
        
        # 障碍物
        self.obstacles = self._parse_obstacles(scene_config.get('obstacles', []))
        
        # 目标生成参数
        target_config = scene_config.get('target_generation', {})
        self.min_distance = target_config.get('min_distance', 5.0)
        self.max_distance = target_config.get('max_distance', 50.0)
        self.min_height = target_config.get('min_height', 3.0)
        self.max_height = target_config.get('max_height', 12.0)
        self.collision_check_samples = target_config.get('collision_check_samples', 100)
        self.max_attempts = target_config.get('max_attempts', 50)
        
        # 碰撞检测网格（加速碰撞检测）
        self.grid_resolution = 1.0  # 网格分辨率（米）
        self.collision_grid = self._build_collision_grid()
        
        self.logger.info(f"Scene manager initialized with {len(self.obstacles)} obstacles")
    
    def _parse_obstacles(self, obstacle_configs: List[Dict[str, Any]]) -> List[Obstacle]:
        """
        解析障碍物配置
        
        Args:
            obstacle_configs: 障碍物配置列表
            
        Returns:
            障碍物对象列表
        """
        obstacles = []
        
        for config in obstacle_configs:
            obstacle_type = config['type']
            name = config.get('name', f"{obstacle_type}_{len(obstacles)}")
            center = np.array(config['center'], dtype=np.float32)
            
            if obstacle_type == 'box':
                size = np.array(config['size'], dtype=np.float32)
            elif obstacle_type == 'cylinder':
                radius = config['radius']
                height = config['height']
                size = np.array([radius, height], dtype=np.float32)
            elif obstacle_type == 'sphere':
                radius = config['radius']
                size = np.array([radius], dtype=np.float32)
            else:
                self.logger.warning(f"Unknown obstacle type: {obstacle_type}")
                continue
            
            obstacle = Obstacle(name=name, type=obstacle_type, center=center, size=size)
            obstacles.append(obstacle)
            
            self.logger.debug(f"Added obstacle: {name} ({obstacle_type}) at {center}")
        
        return obstacles
    
    def _build_collision_grid(self) -> np.ndarray:
        """
        构建碰撞检测网格
        
        Returns:
            碰撞网格（True表示有障碍物）
        """
        # 计算网格尺寸
        x_size = int(np.ceil((self.x_max - self.x_min) / self.grid_resolution))
        y_size = int(np.ceil((self.y_max - self.y_min) / self.grid_resolution))
        z_size = int(np.ceil((self.z_max - self.z_min) / self.grid_resolution))
        
        grid = np.zeros((x_size, y_size, z_size), dtype=bool)
        
        # 填充障碍物到网格
        for obstacle in self.obstacles:
            self._add_obstacle_to_grid(grid, obstacle)
        
        self.logger.debug(f"Built collision grid: {grid.shape}, occupied: {np.sum(grid)} cells")
        
        return grid
    
    def _add_obstacle_to_grid(self, grid: np.ndarray, obstacle: Obstacle):
        """
        将障碍物添加到碰撞网格
        
        Args:
            grid: 碰撞网格
            obstacle: 障碍物对象
        """
        if obstacle.type == 'box':
            self._add_box_to_grid(grid, obstacle)
        elif obstacle.type == 'cylinder':
            self._add_cylinder_to_grid(grid, obstacle)
        elif obstacle.type == 'sphere':
            self._add_sphere_to_grid(grid, obstacle)
    
    def _add_box_to_grid(self, grid: np.ndarray, obstacle: Obstacle):
        """添加盒子障碍物到网格"""
        center = obstacle.center
        size = obstacle.size  # [width, height, depth]
        
        # 计算盒子边界
        x_min = center[0] - size[0] / 2
        x_max = center[0] + size[0] / 2
        y_min = center[1] - size[1] / 2
        y_max = center[1] + size[1] / 2
        z_min = center[2] - size[2] / 2
        z_max = center[2] + size[2] / 2
        
        # 转换为网格坐标
        i_min = max(0, int((x_min - self.x_min) / self.grid_resolution))
        i_max = min(grid.shape[0], int((x_max - self.x_min) / self.grid_resolution) + 1)
        j_min = max(0, int((y_min - self.y_min) / self.grid_resolution))
        j_max = min(grid.shape[1], int((y_max - self.y_min) / self.grid_resolution) + 1)
        k_min = max(0, int((z_min - self.z_min) / self.grid_resolution))
        k_max = min(grid.shape[2], int((z_max - self.z_min) / self.grid_resolution) + 1)
        
        # 填充网格
        grid[i_min:i_max, j_min:j_max, k_min:k_max] = True
    
    def _add_cylinder_to_grid(self, grid: np.ndarray, obstacle: Obstacle):
        """添加圆柱障碍物到网格"""
        center = obstacle.center
        radius = obstacle.size[0]
        height = obstacle.size[1]
        
        # 计算柱体边界
        x_min = center[0] - radius
        x_max = center[0] + radius
        y_min = center[1] - radius  
        y_max = center[1] + radius
        z_min = center[2] - height / 2
        z_max = center[2] + height / 2
        
        # 转换为网格坐标
        i_min = max(0, int((x_min - self.x_min) / self.grid_resolution))
        i_max = min(grid.shape[0], int((x_max - self.x_min) / self.grid_resolution) + 1)
        j_min = max(0, int((y_min - self.y_min) / self.grid_resolution))
        j_max = min(grid.shape[1], int((y_max - self.y_min) / self.grid_resolution) + 1)
        k_min = max(0, int((z_min - self.z_min) / self.grid_resolution))
        k_max = min(grid.shape[2], int((z_max - self.z_min) / self.grid_resolution) + 1)
        
        # 检查每个网格点是否在圆柱内
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    # 网格点的世界坐标
                    x = self.x_min + i * self.grid_resolution
                    y = self.y_min + j * self.grid_resolution
                    z = self.z_min + k * self.grid_resolution
                    
                    # 检查是否在圆柱内
                    dx = x - center[0]
                    dy = y - center[1]
                    distance_2d = np.sqrt(dx*dx + dy*dy)
                    
                    if distance_2d <= radius and z_min <= z <= z_max:
                        grid[i, j, k] = True
    
    def _add_sphere_to_grid(self, grid: np.ndarray, obstacle: Obstacle):
        """添加球体障碍物到网格"""
        center = obstacle.center
        radius = obstacle.size[0]
        
        # 计算球体边界
        x_min = center[0] - radius
        x_max = center[0] + radius
        y_min = center[1] - radius
        y_max = center[1] + radius
        z_min = center[2] - radius
        z_max = center[2] + radius
        
        # 转换为网格坐标
        i_min = max(0, int((x_min - self.x_min) / self.grid_resolution))
        i_max = min(grid.shape[0], int((x_max - self.x_min) / self.grid_resolution) + 1)
        j_min = max(0, int((y_min - self.y_min) / self.grid_resolution))
        j_max = min(grid.shape[1], int((y_max - self.y_min) / self.grid_resolution) + 1)
        k_min = max(0, int((z_min - self.z_min) / self.grid_resolution))
        k_max = min(grid.shape[2], int((z_max - self.z_min) / self.grid_resolution) + 1)
        
        # 检查每个网格点是否在球体内
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    # 网格点的世界坐标
                    x = self.x_min + i * self.grid_resolution
                    y = self.y_min + j * self.grid_resolution
                    z = self.z_min + k * self.grid_resolution
                    
                    # 检查是否在球体内
                    dx = x - center[0]
                    dy = y - center[1]
                    dz = z - center[2]
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if distance <= radius:
                        grid[i, j, k] = True
    
    def generate_target_position(self, start_position: np.ndarray) -> np.ndarray:
        """
        生成科学的目标位置，避免障碍物内和场景外的无效目标
        
        Args:
            start_position: 起始位置
            
        Returns:
            目标位置
        """
        for attempt in range(self.max_attempts):
            # 生成候选目标位置
            target_position = self._generate_candidate_target(start_position)
            
            # 检查是否有效
            if self._is_valid_target_position(target_position, start_position):
                self.logger.debug(f"Generated valid target after {attempt + 1} attempts: {target_position}")
                return target_position
        
        # 如果无法生成有效目标，使用后备策略
        self.logger.warning(f"Failed to generate valid target after {self.max_attempts} attempts, using fallback")
        return self._generate_fallback_target(start_position)
    
    def _generate_candidate_target(self, start_position: np.ndarray) -> np.ndarray:
        """
        生成候选目标位置
        
        Args:
            start_position: 起始位置
            
        Returns:
            候选目标位置
        """
        # 随机生成距离和方向
        distance = np.random.uniform(self.min_distance, self.max_distance)
        
        # 在水平面上随机方向
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 计算水平位置
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        
        # 随机高度
        z = np.random.uniform(self.min_height, self.max_height)
        
        # 候选位置
        candidate = np.array([
            start_position[0] + dx,
            start_position[1] + dy,
            -z  # AirSim使用负Z表示高度
        ], dtype=np.float32)
        
        return candidate
    
    def _is_valid_target_position(self, target_position: np.ndarray, start_position: np.ndarray) -> bool:
        """
        检查目标位置是否有效
        
        Args:
            target_position: 目标位置
            start_position: 起始位置
            
        Returns:
            是否有效
        """
        # 1. 检查是否在场景边界内
        if not self._is_within_bounds(target_position):
            return False
        
        # 2. 检查是否在障碍物内
        if self._is_in_collision(target_position):
            return False
        
        # 3. 检查距离是否合理
        distance = np.linalg.norm(target_position - start_position)
        if distance < self.min_distance or distance > self.max_distance:
            return False
        
        # 4. 检查是否可达（简单的直线可达性检测）
        if not self._is_path_clear(start_position, target_position):
            return False
        
        return True
    
    def _is_within_bounds(self, position: np.ndarray) -> bool:
        """
        检查位置是否在场景边界内
        
        Args:
            position: 位置
            
        Returns:
            是否在边界内
        """
        return (self.x_min <= position[0] <= self.x_max and
                self.y_min <= position[1] <= self.y_max and
                self.z_min <= position[2] <= self.z_max)
    
    def _is_in_collision(self, position: np.ndarray, safety_margin: float = 1.0) -> bool:
        """
        检查位置是否与障碍物碰撞
        
        Args:
            position: 位置
            safety_margin: 安全边距
            
        Returns:
            是否碰撞
        """
        # 使用网格进行快速检测
        grid_x = int((position[0] - self.x_min) / self.grid_resolution)
        grid_y = int((position[1] - self.y_min) / self.grid_resolution)
        grid_z = int((position[2] - self.z_min) / self.grid_resolution)
        
        # 检查网格边界
        if (grid_x < 0 or grid_x >= self.collision_grid.shape[0] or
            grid_y < 0 or grid_y >= self.collision_grid.shape[1] or
            grid_z < 0 or grid_z >= self.collision_grid.shape[2]):
            return True  # 超出网格边界认为碰撞
        
        # 检查周围的网格（考虑安全边距）
        margin_cells = int(np.ceil(safety_margin / self.grid_resolution))
        
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                for dz in range(-margin_cells, margin_cells + 1):
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    check_z = grid_z + dz
                    
                    if (0 <= check_x < self.collision_grid.shape[0] and
                        0 <= check_y < self.collision_grid.shape[1] and
                        0 <= check_z < self.collision_grid.shape[2]):
                        
                        if self.collision_grid[check_x, check_y, check_z]:
                            return True
        
        return False
    
    def _is_path_clear(self, start: np.ndarray, end: np.ndarray) -> bool:
        """
        检查两点之间的直线路径是否无障碍物
        
        Args:
            start: 起始点
            end: 结束点
            
        Returns:
            路径是否清晰
        """
        # 在路径上采样点进行碰撞检测
        num_samples = self.collision_check_samples
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_point = start + t * (end - start)
            
            if self._is_in_collision(sample_point, safety_margin=0.5):
                return False
        
        return True
    
    def _generate_fallback_target(self, start_position: np.ndarray) -> np.ndarray:
        """
        生成后备目标位置（当无法找到理想目标时）
        
        Args:
            start_position: 起始位置
            
        Returns:
            后备目标位置
        """
        # 在起始位置附近生成简单目标
        fallback_distance = min(self.max_distance, 20.0)
        
        # 尝试几个预定义方向
        directions = [
            [1, 0, 0], [-1, 0, 0],  # 东西方向
            [0, 1, 0], [0, -1, 0],  # 南北方向
            [1, 1, 0], [-1, -1, 0],  # 对角方向
            [-1, 1, 0], [1, -1, 0]
        ]
        
        for direction in directions:
            target = start_position + np.array(direction) * fallback_distance
            target[2] = -np.random.uniform(self.min_height, self.max_height)
            
            if self._is_within_bounds(target) and not self._is_in_collision(target):
                self.logger.info(f"Using fallback target: {target}")
                return target
        
        # 最后的后备：在起始位置上方
        fallback_target = start_position.copy()
        fallback_target[2] = -self.max_height
        
        self.logger.warning(f"Using emergency fallback target: {fallback_target}")
        return fallback_target
    
    def get_scene_info(self) -> Dict[str, Any]:
        """
        获取场景信息
        
        Returns:
            场景信息字典
        """
        obstacle_info = []
        for obstacle in self.obstacles:
            info = {
                'name': obstacle.name,
                'type': obstacle.type,
                'center': obstacle.center.tolist(),
                'size': obstacle.size.tolist()
            }
            obstacle_info.append(info)
        
        return {
            'bounds': self.bounds,
            'obstacles': obstacle_info,
            'target_generation': {
                'min_distance': self.min_distance,
                'max_distance': self.max_distance,
                'min_height': self.min_height,
                'max_height': self.max_height
            },
            'collision_grid_shape': self.collision_grid.shape,
            'grid_resolution': self.grid_resolution
        }
    
    def visualize_scene_2d(self, start_position: np.ndarray, target_position: np.ndarray, 
                          height_slice: float = -5.0) -> np.ndarray:
        """
        生成场景的2D可视化（特定高度切片）
        
        Args:
            start_position: 起始位置
            target_position: 目标位置
            height_slice: 高度切片
            
        Returns:
            可视化图像数组
        """
        # 创建图像
        img_width = int((self.x_max - self.x_min) / self.grid_resolution)
        img_height = int((self.y_max - self.y_min) / self.grid_resolution)
        
        # 获取高度切片的网格索引
        z_idx = int((height_slice - self.z_min) / self.grid_resolution)
        if 0 <= z_idx < self.collision_grid.shape[2]:
            slice_grid = self.collision_grid[:, :, z_idx]
        else:
            slice_grid = np.zeros((img_width, img_height), dtype=bool)
        
        # 创建RGB图像
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # 填充障碍物（黑色）
        img[slice_grid.T] = [0, 0, 0]
        
        # 填充自由空间（白色）
        img[~slice_grid.T] = [255, 255, 255]
        
        # 标记起始位置（绿色）
        start_x = int((start_position[0] - self.x_min) / self.grid_resolution)
        start_y = int((start_position[1] - self.y_min) / self.grid_resolution)
        if 0 <= start_x < img_width and 0 <= start_y < img_height:
            img[start_y, start_x] = [0, 255, 0]
        
        # 标记目标位置（红色）
        target_x = int((target_position[0] - self.x_min) / self.grid_resolution)
        target_y = int((target_position[1] - self.y_min) / self.grid_resolution)
        if 0 <= target_x < img_width and 0 <= target_y < img_height:
            img[target_y, target_x] = [255, 0, 0]
        
        return img