import matplotlib.pyplot as plt
from matplotlib import rcParams
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt
from shapely.validation import make_valid
import math
import pandas as pd
import time
from enum import Enum
from matplotlib.patches import Patch
from typing import Dict, Union
from pyproj import Transformer

# ================== 配置部分 ==================
class Direction(Enum):
    """方向枚举（按顺时针顺序）"""
    NORTH = (0, 'north', (337.5, 360), (0, 22.5))
    NORTHEAST = (45, 'northeast', (22.5, 67.5))
    EAST = (90, 'east', (67.5, 112.5))
    SOUTHEAST = (135, 'southeast', (112.5, 157.5))
    SOUTH = (180, 'south', (157.5, 202.5))
    SOUTHWEST = (225, 'southwest', (202.5, 247.5))
    WEST = (270, 'west', (247.5, 292.5))
    NORTHWEST = (315, 'northwest', (292.5, 337.5))
    
    def __init__(self, angle, name, *angle_ranges):
        self.angle = angle
        self._value_ = name
        self.angle_ranges = angle_ranges

class Config:
    """全局配置类"""
    PLOT_SETTINGS = {
        'figsize': (15, 15),
        'font_size': 12,
        'colors': {
            'north': '#FF6B6B',
            'northeast': '#FF9E6B',
            'east': '#4ECDC4',
            'southeast': '#6BD1FF',
            'south': '#87C55F',
            'southwest': '#C5E384',
            'west': '#FFD93D',
            'northwest': '#FFEE9D'
        },
        'alpha': 0.4,
        'linewidth': 0.5
    }
    
    SPLIT_SETTINGS = {
        'sector_step': 0.5,
        'overlap_buffer': 1e-5,
        'coord_precision': 6
    }

# ================== 坐标系转换类 ==================
class CoordinateConverter:
    """坐标系转换工具类"""
    def __init__(self):
        self.gcj2wgs = Transformer.from_crs("EPSG:4490", "EPSG:4326")
        
    def bd09_to_wgs84(self, lon: float, lat: float) -> tuple:
        """BD09转WGS84（高精度版）"""
        x_pi = math.pi * 3000.0 / 180.0
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        raw_lon, raw_lat = self.gcj2wgs.transform(gcj_lon, gcj_lat)
        return round(raw_lon, 6), round(raw_lat, 6)  # 立即四舍五入到6位小数

# ================== 几何处理核心类 ==================
class GeometryProcessor:
    """几何处理工具集"""
    
    @staticmethod
    def create_polygon(source: Union[str, list, Polygon]) -> Polygon:
        """创建验证后的多边形"""
        if isinstance(source, str):
            geom = wkt.loads(source)
        elif isinstance(source, (list, np.ndarray)):
            # 输入时立即应用精度控制
            rounded = np.round(source, Config.SPLIT_SETTINGS['coord_precision'])
            geom = Polygon(rounded)
        elif isinstance(source, Polygon):
            geom = source
        else:
            raise TypeError("不支持的输入类型")
        
        return GeometryProcessor.validate_polygon(geom)
    
    @staticmethod
    def validate_polygon(poly: Polygon) -> Polygon:
        """几何验证与修复"""
        if not poly.is_valid:
            poly = make_valid(poly.buffer(0))
        if poly.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError("无效的几何类型")
        return poly
    
    @staticmethod
    def parse_ring_polygon(wkt_str: str) -> tuple:
        """解析环状多边形为外环和内环"""
        poly = GeometryProcessor.create_polygon(wkt_str)
        
        if len(poly.interiors) == 0:
            raise ValueError("输入多边形必须包含至少一个内环形成环状结构")
        if len(poly.interiors) > 1:
            print("警告: 检测到多个内环，默认使用第一个内环")

        outer = Polygon(poly.exterior)
        inner = Polygon(poly.interiors[0])
        
        if not inner.within(outer):
            raise ValueError("内环必须完全在外环内部")
            
        return outer, inner

# ================== 方位分割器 ==================
class QuadrantSplitter:
    """8方向区域分割器（支持单环输入）"""
    
    def __init__(self, ring_wkt: str):
        self.outer, self.inner = GeometryProcessor.parse_ring_polygon(ring_wkt)
        self.centroid = self.inner.centroid
    
    def split_zones(self) -> Dict[str, Polygon]:
        """执行方位分割"""
        ring_zone = self.outer.difference(self.inner)
        ring_zone = GeometryProcessor.validate_polygon(ring_zone)
        
        radius = self._calculate_max_radius(ring_zone)
        subzones = {}
        
        for direction in Direction:
            sector = self._create_sector(direction.angle_ranges, radius)
            intersection = ring_zone.intersection(sector)
            subzones[direction.value] = make_valid(intersection) if not intersection.is_empty else None
        
        self._validate_result(subzones, ring_zone)
        return subzones
    
    def _calculate_max_radius(self, zone: Polygon) -> float:
        """计算最大分割半径"""
        bounds = zone.bounds
        dx = max(abs(bounds[2] - self.centroid.x), abs(bounds[0] - self.centroid.x))
        dy = max(abs(bounds[3] - self.centroid.y), abs(bounds[1] - self.centroid.y))
        return math.hypot(dx, dy) * 1.2
    
    def _create_sector(self, angle_ranges: tuple, radius: float) -> Polygon:
        """创建扇形区域"""
        step = Config.SPLIT_SETTINGS['sector_step']
        points = [(self.centroid.x, self.centroid.y)]
        
        for angle_range in angle_ranges:
            start, end = angle_range
            angles = np.arange(start, end + step, step) if start < end else \
                     np.concatenate([np.arange(start, 360, step), np.arange(0, end + step, step)])
            
            math_angles = (90 - angles) % 360
            radians = np.deg2rad(math_angles)
            
            for angle_rad in radians:
                x = self.centroid.x + radius * np.cos(angle_rad)
                y = self.centroid.y + radius * np.sin(angle_rad)
                points.append((
                    round(x, Config.SPLIT_SETTINGS['coord_precision']),  # 应用精度控制
                    round(y, Config.SPLIT_SETTINGS['coord_precision'])
                ))
        
        return Polygon(points).buffer(Config.SPLIT_SETTINGS['overlap_buffer'], join_style=2)
    
    def _validate_result(self, subzones: dict, original_zone: Polygon):
        """验证结果完整性"""
        total_area = sum(geom.area for geom in subzones.values() if geom)
        if abs(total_area - original_zone.area) > max(1.0, original_zone.area * 0.01):
            raise ValueError("区域分割完整性验证失败")

# ================== 可视化与输出类 ==================
class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, splitter: QuadrantSplitter):
        self.splitter = splitter
    
    def plot_base(self):
        """绘制基础环形区域"""
        fig, ax = self._create_figure()
        
        # 绘制环形区域
        gpd.GeoSeries(self.splitter.outer.difference(self.splitter.inner)).plot(
            ax=ax, color='#f0f0f0', alpha=0.3, edgecolor='k')
        
        # 绘制边界
        gpd.GeoSeries([self.splitter.outer, self.splitter.inner]).boundary.plot(
            ax=ax, color='grey', linewidth=1)
        
        # 标记质心
        ax.scatter(*self.splitter.centroid.coords[0], s=200, c='purple',
                  marker='*', edgecolor='white', label='质心')
        
        plt.title("基础环形区域", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_quadrants(self, subzones: dict):
        """绘制方位分割结果"""
        fig, ax = self._create_figure()
        colors = Config.PLOT_SETTINGS['colors']
        
        # 绘制各方向区域
        for dir_name, geom in subzones.items():
            if geom and not geom.is_empty:
                gpd.GeoSeries(geom).plot(
                    ax=ax, 
                    color=colors[dir_name], 
                    alpha=Config.PLOT_SETTINGS['alpha'],
                    edgecolor='k',
                    linewidth=Config.PLOT_SETTINGS['linewidth']
                )
                
                # 添加方向标注
                try:
                    part_centroid = geom.centroid
                    ax.text(part_centroid.x, part_centroid.y, 
                           dir_name.upper(),  
                           ha='center', va='center',
                           fontsize=10, fontweight='bold')
                except:
                    pass
        
        # 绘制方向指示线
        radius = self._calculate_visual_radius(ax)
        self._draw_direction_lines(ax, radius)
        self._add_legend(ax)
        plt.title("8方向方位分割结果", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _create_figure(self):
        """创建基础绘图对象"""
        fig, ax = plt.subplots(figsize=Config.PLOT_SETTINGS['figsize'])
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        rcParams.update({'font.size': Config.PLOT_SETTINGS['font_size']})
        return fig, ax
    
    def _calculate_visual_radius(self, ax):
        """计算可视化半径"""
        x_radius = ax.get_xlim()[1] - self.splitter.centroid.x
        y_radius = ax.get_ylim()[1] - self.splitter.centroid.y
        return max(x_radius, y_radius) * 0.7
    
    def _draw_direction_lines(self, ax, radius):
        """绘制方向指示线"""
        for direction in Direction:
            math_angle = (90 - direction.angle) % 360
            angle_rad = math.radians(math_angle)
            end_x = self.splitter.centroid.x + radius * np.cos(angle_rad)
            end_y = self.splitter.centroid.y + radius * np.sin(angle_rad)
            ax.plot([self.splitter.centroid.x, end_x], 
                    [self.splitter.centroid.y, end_y], 
                    'k--', alpha=0.5, linewidth=0.5)
    
    def _add_legend(self, ax):
        """添加图例"""
        colors = Config.PLOT_SETTINGS['colors']
        legend_elements = [Patch(facecolor=colors[d.value], label=d.value.upper()) 
                          for d in Direction]
        ax.legend(handles=legend_elements, loc='upper right', ncol=2)

class DataExporter:
    """数据导出器（支持双坐标系）"""
    
    def __init__(self):
        self.converter = CoordinateConverter()
    
    def export(self, subzones: dict, filename: str):
        """导出分区数据到CSV"""
        data = []
        for dir_name, geom in subzones.items():
            if not geom or geom.is_empty:
                continue
                
            if isinstance(geom, MultiPolygon):
                for i, part in enumerate(geom.geoms):
                    data.append(self._create_record(dir_name, i+1, part))
            else:
                data.append(self._create_record(dir_name, 1, geom))
        
        pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已导出至: {filename}")

    def _create_record(self, dir_name: str, part_id: int, geom: Polygon) -> dict:
        """创建双坐标系数据记录"""
        return {
            'direction': dir_name,
            'part_id': part_id,
            'area_m2': round(geom.area, 2),
            'wkt_bd09': self._clean_geometry(geom),
            'wkt_wgs84': self._convert_geometry(geom)
        }

    def _clean_geometry(self, geom: Polygon) -> str:
        """清洗BD09几何坐标"""
        def process_ring(ring):
            # 使用numpy进行坐标舍入和去重
            arr = np.round(ring.coords, Config.SPLIT_SETTINGS['coord_precision'])
            _, idx = np.unique(arr, axis=0, return_index=True)
            unique = arr[np.sort(idx)]
            if not np.array_equal(unique[0], unique[-1]):
                unique = np.vstack([unique, unique[0]])
            return unique
        
        exterior = process_ring(geom.exterior)
        interiors = [process_ring(interior) for interior in geom.interiors]
        return Polygon(exterior, interiors).wkt

    def _convert_geometry(self, geom: Polygon) -> str:
        """转换坐标系到WGS84并保留6位小数"""
        def convert_ring(ring):
            converted = []
            for x, y in ring.coords:
                # 转换坐标并四舍五入到6位小数
                lon, lat = self.converter.bd09_to_wgs84(x, y)
                converted.append((lon, lat))
            return converted
        
        try:
            # 处理外环坐标
            exterior = convert_ring(geom.exterior)
            # 处理内环坐标
            interiors = [convert_ring(interior) for interior in geom.interiors]
            
            # 重建几何体并确保闭合
            processed = Polygon(exterior, interiors)
            processed = make_valid(processed)
            
            # 二次精度验证
            def clamp_ring(ring):
                return [(round(x,6), round(y,6)) for x, y in ring.coords]
            
            exterior_clamped = clamp_ring(processed.exterior)
            interiors_clamped = [clamp_ring(ring) for ring in processed.interiors]
            
            return Polygon(exterior_clamped, interiors_clamped).wkt
        except Exception as e:
            print(f"坐标转换失败: {str(e)}")
            return 'GEOMETRY_ERROR'

# ================== 主程序 ==================
if __name__ == "__main__":
    RING_WKT = """"""  # 输入环状polygon的WKT

    try:
        # 初始化组件
        splitter = QuadrantSplitter(RING_WKT)
        visualizer = ResultVisualizer(splitter)
        exporter = DataExporter()
        
        # 可视化基础结构
        visualizer.plot_base()
        
        # 执行方位分割
        print("开始方位分割...")
        start_time = time.time()
        subzones = splitter.split_zones()
        print(f"分割完成，耗时: {time.time()-start_time:.2f}秒")
        
        # 打印统计信息
        total_area = splitter.outer.area - splitter.inner.area
        print(f"\n总环形面积: {total_area:.0f}m²")
        for direction in Direction:
            dir_name = direction.value
            area = subzones[dir_name].area if subzones[dir_name] else 0
            print(f"  {dir_name}: {area:.0f}m² ({area/total_area*100:.1f}%)")
        
        # 可视化结果
        visualizer.plot_quadrants(subzones)
        
        # 导出数据
        exporter.export(subzones, 'direction_zones.csv')
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
