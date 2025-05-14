import matplotlib.pyplot as plt
from matplotlib import rcParams
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List
import time
from enum import Enum
from matplotlib.patches import Patch
from shapely import wkt
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
            Direction.NORTH.value: '#FF6B6B',
            Direction.NORTHEAST.value: '#FF9E6B',
            Direction.EAST.value: '#4ECDC4',
            Direction.SOUTHEAST.value: '#6BD1FF',
            Direction.SOUTH.value: '#87C55F',
            Direction.SOUTHWEST.value: '#C5E384',
            Direction.WEST.value: '#FFD93D',
            Direction.NORTHWEST.value: '#FFEE9D'
        },
        'ring_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'],  # 增加外接AOI颜色
        'alpha': 0.4,
        'linewidth': 0.5,
        'hatch': ['//', '\\\\', '||', 'xx'],
    }
    
    SPLIT_SETTINGS = {
        'num_rays': 180,
        'sector_step': 0.5,
        'buffer_size': 0.05,
        'min_sector_area': 1.0,
        'overlap_buffer': 1e-5
    }

def init_environment():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    rcParams.update({'font.size': Config.PLOT_SETTINGS['font_size']})

# ================== 坐标系转换类 ==================
class CoordinateConverter:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:4490", "EPSG:4326", always_xy=True)

    def bd09_to_wgs84(self, lon: float, lat: float) -> Tuple[float, float]:
        """BD09坐标系转WGS84坐标系"""
        x_pi = math.pi * 3000.0 / 180.0
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        wgs_lon, wgs_lat = self.transformer.transform(gcj_lon, gcj_lat)
        return round(wgs_lon, 6), round(wgs_lat, 6)

    def convert_geometry(self, geom):
        """递归转换几何体坐标系"""
        if geom is None or geom.is_empty:
            return None
        
        if isinstance(geom, Polygon):
            # 转换外环坐标
            exterior = geom.exterior
            converted_exterior = [self.bd09_to_wgs84(*p) for p in exterior.coords]
            
            # 转换内环坐标（孔洞）
            interiors = []
            for interior in geom.interiors:
                converted_interior = [self.bd09_to_wgs84(*p) for p in interior.coords]
                interiors.append(converted_interior)
            
            return Polygon(converted_exterior, interiors)
        
        elif isinstance(geom, MultiPolygon):
            # 转换每个子多边形
            return MultiPolygon([
                self.convert_geometry(poly) 
                for poly in geom.geoms
            ])
        
        return geom

# ================== 几何处理核心类 ==================
class GeometryValidator:
    @staticmethod
    def validate_polygon(poly: Polygon) -> Polygon:
        if not poly.is_valid:
            poly = make_valid(poly)
            if not poly.is_valid:
                raise ValueError("无法修复无效的多边形")
        return poly
    
    @staticmethod
    def ensure_multipolygon(geom) -> MultiPolygon:
        if isinstance(geom, MultiPolygon):
            return geom
        elif isinstance(geom, Polygon):
            return MultiPolygon([geom])
        else:
            raise ValueError(f"不支持的几何类型: {type(geom)}")

class RingDivider:
    def __init__(self, polygon: Polygon):
        self.polygon = GeometryValidator.validate_polygon(polygon)
        self.centroid = self.polygon.centroid
        self._validate_geometry()

    def _validate_geometry(self):
        if not self.polygon.is_valid:
            raise ValueError("无效的几何图形，请检查坐标数据")

    def calculate_total_area(self) -> float:
        return self.polygon.area

    def divide_into_three_parts(self) -> Dict[str, Polygon]:
        start_time = time.time()
        
        angles = np.linspace(0, 2 * np.pi, Config.SPLIT_SETTINGS['num_rays'], endpoint=False)
        intersection_points = self._calculate_intersections(angles)
        division_points = self._calculate_division_points(intersection_points)
        parts = self._create_parts(division_points)
        
        print(f"三等分完成，耗时: {time.time()-start_time:.2f}秒")
        return parts

    def _calculate_intersections(self, angles: np.ndarray) -> List[Tuple[Point, Point]]:
        intersection_points = []
        for angle in angles:
            ray = self._create_ray(angle)
            pts = self._get_intersections(ray, self.polygon.exterior)
            if len(pts) < 2:
                pts = [Point(self.centroid.x, self.centroid.y)] + pts
            intersection_points.append(pts)
        return intersection_points

    def _create_ray(self, angle: float) -> LineString:
        return LineString([
            (self.centroid.x, self.centroid.y),
            (self.centroid.x + 100 * np.cos(angle),
             self.centroid.y + 100 * np.sin(angle))
        ])

    def _get_intersections(self, ray: LineString, boundary) -> List[Point]:
        intersection = ray.intersection(boundary)
        if intersection.is_empty:
            return []
        if intersection.geom_type == 'MultiPoint':
            points = [p for p in intersection.geoms]
            points.sort(key=lambda p: p.distance(self.centroid))
            return points
        return [intersection]

    def _calculate_division_points(self, intersection_points: List[List[Point]]) -> List[List[Tuple[float, float]]]:
        division_points = []
        for pts in intersection_points:
            if len(pts) < 2:
                inner_pt = Point(self.centroid.x, self.centroid.y)
                outer_pt = pts[0] if pts else inner_pt
            else:
                inner_pt, outer_pt = pts[0], pts[-1]
            
            points = []
            for d in range(4):
                t = d / 3
                x = inner_pt.x + t * (outer_pt.x - inner_pt.x)
                y = inner_pt.y + t * (outer_pt.y - inner_pt.y)
                points.append((x, y))
            division_points.append(points)
        return division_points

    def _create_parts(self, division_points: List[List[Tuple[float, float]]]) -> Dict[str, Polygon]:
        parts = {}
        rings = []
        for i in range(3):
            coords = [division_points[j][i+1] for j in range(len(division_points))]
            coords.append(division_points[0][i+1])
            rings.append(GeometryValidator.validate_polygon(Polygon(coords)))
        
        parts['part1'] = rings[0]
        parts['part2'] = rings[1].difference(rings[0])
        parts['part3'] = self.polygon.difference(rings[1])
        return parts

    def create_external_aoi(self, extension_factor: float = 1/3) -> Dict[str, Polygon]:
        """创建外接AOI区域（默认扩展1/3长度）"""
        start_time = time.time()
        
        angles = np.linspace(0, 2 * np.pi, Config.SPLIT_SETTINGS['num_rays'], endpoint=False)
        intersection_points = self._calculate_intersections(angles)
        
        # 计算外边界点
        external_points = []
        for pts in intersection_points:
            if len(pts) < 2:
                inner_pt = Point(self.centroid.x, self.centroid.y)
                outer_pt = pts[0] if pts else inner_pt
            else:
                inner_pt, outer_pt = pts[0], pts[-1]
            
            # 计算扩展后的点
            extension_length = inner_pt.distance(outer_pt) * extension_factor
            angle = math.atan2(outer_pt.y - inner_pt.y, outer_pt.x - inner_pt.x)
            
            extended_x = outer_pt.x + extension_length * math.cos(angle)
            extended_y = outer_pt.y + extension_length * math.sin(angle)
            external_points.append((extended_x, extended_y))
        
        # 创建外部多边形
        if len(external_points) < 3:
            return {'external_aoi': None}
        
        external_poly = Polygon(external_points)
        aoi_ring = external_poly.difference(self.polygon)
        
        # 验证结果
        valid_aoi = GeometryValidator.validate_polygon(aoi_ring)
        result = {
            'original': self.polygon,
            'external_aoi': valid_aoi if not valid_aoi.is_empty else None,
            'extension_factor': extension_factor
        }
        
        print(f"外接AOI区域创建完成，耗时: {time.time()-start_time:.2f}秒")
        return result

class QuadrantSplitter:
    @staticmethod
    def split_zone(zone: Polygon, centroid: Point, config: dict = None) -> Dict[str, Optional[Polygon]]:
        config = config or Config.SPLIT_SETTINGS
        
        zone = GeometryValidator.validate_polygon(zone)
        if zone.is_empty:
            return {d.value: None for d in Direction}
        
        radius = QuadrantSplitter._calculate_max_radius(zone, centroid)
        subzones = {}
        
        for direction in Direction:
            sector = QuadrantSplitter._create_precise_sector(
                centroid,
                direction.angle_ranges,
                radius
            )
            
            if sector.is_empty:
                subzones[direction.value] = None
                continue
            
            intersection = zone.intersection(sector)
            if intersection.is_empty:
                subzones[direction.value] = None
            else:
                valid_geom = make_valid(intersection)
                subzones[direction.value] = valid_geom if not valid_geom.is_empty else None
        
        QuadrantSplitter._validate_subzones(subzones, zone)
        return subzones

    @staticmethod
    def _calculate_max_radius(zone: Polygon, centroid: Point) -> float:
        bounds = zone.bounds
        dx = max(abs(bounds[2] - centroid.x), abs(bounds[0] - centroid.x))
        dy = max(abs(bounds[3] - centroid.y), abs(bounds[1] - centroid.y))
        return math.hypot(dx, dy) * 2.0

    @staticmethod
    def _create_precise_sector(center: Point, angle_ranges: tuple, radius: float) -> Polygon:
        step = Config.SPLIT_SETTINGS['sector_step']
        points = [(center.x, center.y)]
        
        for angle_range in angle_ranges:
            start, end = angle_range
            if start > end:
                angles = np.concatenate([
                    np.arange(start, 360, step),
                    np.arange(0, end + step, step)
                ])
            else:
                angles = np.arange(start, end + step, step)
            
            math_angles = (90 - angles) % 360
            radians = np.deg2rad(math_angles)
            
            for angle_rad in radians:
                dx = radius * np.cos(angle_rad)
                dy = radius * np.sin(angle_rad)
                points.append((center.x + dx, center.y + dy))
        
        if len(points) < 3:
            return Polygon()
        
        sector = Polygon(points).buffer(
            Config.SPLIT_SETTINGS['overlap_buffer'],
            join_style=2
        )
        return GeometryValidator.validate_polygon(sector)

    @staticmethod
    def _validate_subzones(subzones: Dict[str, Optional[Polygon]], zone: Polygon):
        total_area = 0
        valid_geoms = []
        
        for dir, geom in subzones.items():
            if geom is None:
                continue
            total_area += geom.area
            valid_geoms.append(geom)
        
        original_area = zone.area
        tolerance = max(1.0, original_area * 0.01)
        if abs(total_area - original_area) > tolerance:
            print(f"警告: 子区域总面积不匹配 (原始面积: {original_area:.2f}, 子区域总面积: {total_area:.2f})")
        
        if len(valid_geoms) > 1:
            union = unary_union(valid_geoms)
            if union.area > total_area + tolerance:
                print(f"警告: 子区域可能存在重叠 (并集面积: {union.area:.2f}, 子区域总面积: {total_area:.2f})")

# ================== 可视化与输出类 ==================
class Visualizer:
    @staticmethod
    def plot_parts(polygon: Polygon, parts: Dict[str, Polygon], centroid: Point):
        fig, ax = plt.subplots(figsize=Config.PLOT_SETTINGS['figsize'])
        
        for i, (name, part) in enumerate(parts.items()):
            gpd.GeoSeries(part).plot(
                ax=ax, 
                color=Config.PLOT_SETTINGS['ring_colors'][i],
                alpha=Config.PLOT_SETTINGS['alpha'],
                label=name,
                hatch=Config.PLOT_SETTINGS['hatch'][i]
            )
        
        gpd.GeoSeries(polygon).boundary.plot(
            ax=ax, color='grey', linewidth=1)
        
        ax.scatter(centroid.x, centroid.y, s=200, c='purple',
                  marker='*', edgecolor='white', label='质心')
        
        plt.title("区域划分", fontsize=14)
        
        legend_patches = [
            Patch(facecolor=Config.PLOT_SETTINGS['ring_colors'][i], 
                  label=name, hatch=Config.PLOT_SETTINGS['hatch'][i])
            for i, name in enumerate(parts.keys())
        ]
        legend_patches.append(
            plt.Line2D([0], [0], marker='*', color='w', label='质心',
                      markerfacecolor='purple', markersize=15)
        )
        plt.legend(handles=legend_patches)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_quadrants(subzones_dict: Dict[str, Dict[str, Polygon]], 
                      centroid: Point):
        fig, ax = plt.subplots(figsize=Config.PLOT_SETTINGS['figsize'])
        colors = Config.PLOT_SETTINGS['colors']
        
        for zone_name, subzones in subzones_dict.items():
            for dir, geom in subzones.items():
                if geom and not geom.is_empty:
                    gpd.GeoSeries(geom).plot(
                        ax=ax, 
                        color=colors[dir], 
                        alpha=Config.PLOT_SETTINGS['alpha'],
                        edgecolor='k',
                        linewidth=Config.PLOT_SETTINGS['linewidth'],
                        label=f'{zone_name}_{dir}'
                    )
                    try:
                        part_centroid = geom.centroid
                        ax.text(part_centroid.x, part_centroid.y,
                               f"{dir}\n",                      
                    #面积不准           f"{dir}\n{geom.area:.0f}m²", 
                               ha='center', va='center',
                               fontsize=8, fontweight='bold')
                    except:
                        pass
        
        radius = max(
            abs(ax.get_xlim()[1] - centroid.x),
            abs(ax.get_ylim()[1] - centroid.y)
        ) * 0.7
        
        for direction in Direction:
            math_angle = (90 - direction.angle) % 360
            math_angle_rad = math.radians(math_angle)
            
            end_x = centroid.x + radius * math.cos(math_angle_rad)
            end_y = centroid.y + radius * math.sin(math_angle_rad)
            
            ax.plot([centroid.x, end_x], [centroid.y, end_y],
                   'k--', linewidth=0.5, alpha=0.5)
        
        legend_elements = [
            Patch(facecolor=colors[d.value], label=d.value.upper())
            for d in Direction
        ]
        ax.legend(handles=legend_elements, loc='upper right', ncol=2)
        plt.title("方位区域切分结果 (8方向)", fontsize=14)
        plt.tight_layout()
        plt.show()

class DataExporter:
    converter = CoordinateConverter()  # 实例化坐标转换器

    @staticmethod
    def _format_wkt(geom, convert: bool = False):
        """
        生成WKT字符串
        :param convert: 是否转换为WGS84坐标系
        """
        if geom is None or geom.is_empty:
            return ""
        
        # 坐标转换处理
        target_geom = DataExporter.converter.convert_geometry(geom) if convert else geom
        
        # 生成WKT并格式化
        import re
        wkt_str = target_geom.wkt
        wkt_str = re.sub(r"(\d+\.\d{6})\d+", r"\1", wkt_str)  # 保留6位小数
        return wkt_str.replace('\n', ' ').replace('\r', ' ')

    @staticmethod
    def export_parts(parts: Dict[str, Polygon], filename: str):
        data = []
        for name, part in parts.items():
            if isinstance(part, MultiPolygon):
                for i, p in enumerate(part.geoms):
                    data.append({
                        'part': name,
                        'part_id': i+1,
                        'area_m2': p.area,
                        'wkt_bd09': DataExporter._format_wkt(p),  # 原始坐标系
                        'wkt_wgs84': DataExporter._format_wkt(p, convert=True)  # 转换后坐标系
                    })
            else:
                data.append({
                    'part': name,
                    'part_id': 1,
                    'area_m2': part.area,
                    'wkt_bd09': DataExporter._format_wkt(part),
                    'wkt_wgs84': DataExporter._format_wkt(part, convert=True)
                })
        
        pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"区域数据已导出到: {filename} (包含BD09/WGS84双坐标系)")

    @staticmethod
    def export_subzones(subzones_dict: Dict[str, Dict[str, Polygon]], 
                       filename: str):
        data = []
        for zone_name, subzones in subzones_dict.items():
            for dir, geom in subzones.items():
                if geom is None or geom.is_empty:
                    continue
                
                if isinstance(geom, MultiPolygon):
                    for i, part in enumerate(geom.geoms):
                        data.append({
                            'zone': zone_name,
                            'direction': dir,
                            'part_id': i+1,
                            'area_m2': part.area,
                            'wkt_bd09': DataExporter._format_wkt(part),
                            'wkt_wgs84': DataExporter._format_wkt(part, convert=True)
                        })
                else:
                    data.append({
                        'zone': zone_name,
                        'direction': dir,
                        'part_id': 1,
                        'area_m2': geom.area,
                        'wkt_bd09': DataExporter._format_wkt(geom),
                        'wkt_wgs84': DataExporter._format_wkt(geom, convert=True)
                    })
        
        pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"子区域数据已导出到: {filename} (包含BD09/WGS84双坐标系)")

# ================== 主程序 ==================
if __name__ == "__main__":
    # 初始化环境
    init_environment()
    
    # 解析WKT,以北京二环为例，这里使用的二环是封闭的polygon
    erhuan_wkt = ""
    erhuan_poly = wkt.loads(erhuan_wkt)
    
    # 创建分割器
    divider = RingDivider(erhuan_poly)
    
    # 计算总面积
    total_area = divider.calculate_total_area()
    print(f"二环总面积: {total_area:.2f} 平方米")
    print(f"目标每个分区面积: {total_area/3:.2f} 平方米")
    
    # 三等分区域
    parts = divider.divide_into_three_parts()
    
    # 计算实际面积
    print("\n实际分区面积:")
    for name, part in parts.items():
        area = part.area
        print(f"{name}: {area:.2f} 平方米 ({area/total_area*100:.2f}%)")
    
    
    # 可视化所有区域
    Visualizer.plot_parts(divider.polygon, parts, divider.centroid)
    
    # 方位分割（8方向）- 对所有区域进行分割
    print("\n开始方位分割...")
    start_time = time.time()
    subzones_all = {
        name: QuadrantSplitter.split_zone(part, divider.centroid)
        for name, part in parts.items()
    }
    print(f"方位分割完成，耗时: {time.time()-start_time:.2f}秒")
    
    # 打印统计信息
    print("\n区域面积统计：")
    for part_name, part in parts.items():
        total = part.area
        print(f"\n[{part_name}] 总面积：{total:.0f}m²")
        for direction in Direction:
            dir = direction.value
            area = subzones_all[part_name][dir].area if subzones_all[part_name][dir] else 0
            print(f"  {dir}: {area:.0f}m² ({area/total*100:.1f}%)")
    
    # 可视化方位分割结果
    Visualizer.plot_quadrants(subzones_all, divider.centroid)
    
    # 导出数据（包含双坐标系）
    DataExporter.export_parts(parts, 'erhuan_parts.csv')
    DataExporter.export_subzones(subzones_all, 'erhuan_direction_subzones.csv')
