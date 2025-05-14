import matplotlib.pyplot as plt
from matplotlib import rcParams
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon, Point, LinearRing, MultiPoint
from shapely import wkt, validation, ops
import pandas as pd
from shapely.ops import unary_union, nearest_points
import math
from pyproj import Transformer
from typing import Tuple, List, Union
import csv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 改用宋体（支持中文+上标）
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先SimHei，缺失字符用DejaVu
# ================== 增强版坐标系转换模块 ==================
class CoordinateConverter:
    def __init__(self):
        # 初始化转换器（使用中国官方参数）
        self.gcj02_to_wgs84 = Transformer.from_crs("EPSG:4490", "EPSG:4326", always_xy=True)
        self.wgs84_to_gcj02 = Transformer.from_crs("EPSG:4326", "EPSG:4490", always_xy=True)
        
    def bd09_to_wgs84(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        BD09转WGS84坐标系（高精度版）
        算法步骤：BD09 → GCJ02 → WGS84
        """
        # BD09转GCJ02
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x*x + y*y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        
        # GCJ02转WGS84
        wgs_lon, wgs_lat = self.gcj02_to_wgs84.transform(gcj_lon, gcj_lat)
        return round(wgs_lon, 6), round(wgs_lat, 6)

    def wgs84_to_bd09(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        WGS84转BD09坐标系（高精度版）
        算法步骤：WGS84 → GCJ02 → BD09
        """
        # WGS84转GCJ02
        gcj_lon, gcj_lat = self.wgs84_to_gcj02.transform(lon, lat)
        
        # GCJ02转BD09
        z = math.sqrt(gcj_lon*gcj_lon + gcj_lat*gcj_lat) + 0.00002 * math.sin(gcj_lat * self.x_pi)
        theta = math.atan2(gcj_lat, gcj_lon) + 0.000003 * math.cos(gcj_lon * self.x_pi)
        bd_lon = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return round(bd_lon, 6), round(bd_lat, 6)

    def convert_geometry(self, geom: Union[Polygon, MultiPolygon], to_wgs84: bool = True) -> Union[Polygon, MultiPolygon]:
        """
        几何对象坐标系转换（支持批量处理）
        :param geom: 输入几何对象
        :param to_wgs84: True表示转WGS84，False表示转BD09
        :return: 转换后的几何对象
        """
        if geom is None or geom.is_empty:
            return geom
            
        def transform_coords(coords):
            if to_wgs84:
                return [self.bd09_to_wgs84(*c) for c in coords]
            else:
                return [self.wgs84_to_bd09(*c) for c in coords]
        
        if isinstance(geom, Polygon):
            # 转换外环
            new_exterior = transform_coords(geom.exterior.coords)
            # 转换内环
            new_interiors = [transform_coords(interior.coords) for interior in geom.interiors]
            return GeometryValidator.validate_geometry(Polygon(new_exterior, new_interiors))
            
        elif isinstance(geom, MultiPolygon):
            return MultiPolygon([self.convert_geometry(p, to_wgs84) for p in geom.geoms])
            
        return geom

    @property
    def x_pi(self):
        return math.pi * 3000.0 / 180.0  # BD09专用参数

# ================== 几何体验证模块（保持不变） ==================
class GeometryValidator:
    @staticmethod
    def ensure_closed_ring(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """确保环闭合（首尾点相同）"""
        if len(coords) < 4:
            return coords
        if coords[0] != coords[-1]:
            return coords + [coords[0]]
        return coords

    @staticmethod
    def validate_geometry(geom):
        """几何体修复验证"""
        if geom.is_valid:
            return geom
        try:
            repaired = validation.make_valid(geom)
            if repaired.geom_type == 'GeometryCollection':
                polygons = [g for g in repaired.geoms if g.geom_type == 'Polygon']
                return max(polygons, key=lambda g: g.area) if polygons else geom
            return repaired
        except:
            return geom

# ================== WKT生成模块（保持不变） ==================
class WKTGenerator:
    @staticmethod
    def create_polygon(exterior: List[Tuple[float, float]], 
                      interiors: List[List[Tuple[float, float]]] = None) -> Polygon:
        """创建带孔洞的多边形"""
        exterior_ring = GeometryValidator.ensure_closed_ring(exterior)
        interior_rings = [GeometryValidator.ensure_closed_ring(ring) for ring in (interiors or [])]
        return Polygon(exterior_ring, interior_rings)

# ================== 主程序逻辑 ==================
# 配置显示参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 12})

# 初始化组件
converter = CoordinateConverter()

def format_coords(coords):
    """坐标格式化（兼容旧代码）"""
    return [(round(x, 6), round(y, 6)) for x, y in coords]

def get_single_point(intersection, centroid):
    """获取有效点（增强容错）"""
    try:
        if isinstance(intersection, MultiPoint):
            return nearest_points(intersection, centroid)[0]
        return intersection if isinstance(intersection, Point) else centroid
    except:
        return centroid

# 示例数据，环状多边形polygon的WKT字符串
sample_wkt = """"""  # 保持原样

# 数据预处理
polygon = wkt.loads(sample_wkt)
outer_ring = LinearRing(format_coords(polygon.exterior.coords))
inner_ring = LinearRing(format_coords(polygon.interiors[0].coords)) if polygon.interiors else None
centroid = outer_ring.centroid

# 生成射线交点（保持不变）
num_rays = 180
angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
intersections = []
for angle in angles:
    ray = LineString([
        centroid, 
        (centroid.x + 100*np.cos(angle), centroid.y + 100*np.sin(angle))
    ])
    inner_pt = get_single_point(ray.intersection(inner_ring), centroid) if inner_ring else centroid
    outer_pt = get_single_point(ray.intersection(outer_ring), centroid)
    intersections.append((inner_pt, outer_pt))

def create_ring_zone(pts):
    """创建闭合环状区域（增强容错）"""
    coords = [(round(pt.x,6), round(pt.y,6)) for pt in pts]  # 坐标精度处理
    return GeometryValidator.validate_geometry(LinearRing(GeometryValidator.ensure_closed_ring(coords)))

# 计算四段式环带（新增扩展环）
ring_pts = [[] for _ in range(3)]  # 0:1/3处, 1:2/3处, 2:4/3处（外扩）
for inner_pt, outer_pt in intersections:
    # 计算基础向量
    dx = outer_pt.x - inner_pt.x
    dy = outer_pt.y - inner_pt.y
    
    # 生成三个分界点
    for i, ratio in enumerate([1/3, 2/3, 4/3], start=1):
        x = inner_pt.x + ratio * dx
        y = inner_pt.y + ratio * dy
        ring_pts[i-1].append(Point(x, y))

# 构建环状分界线
split_rings = [create_ring_zone(pts) for pts in ring_pts]

# 构建四个分区（新增扩展环带）
zones = []
if inner_ring:
    zones.append(Polygon(split_rings[0].coords, [inner_ring.coords]))  # 内环区
else:
    zones.append(Polygon(split_rings[0].coords))
    
zones.extend([
    Polygon(split_rings[1].coords, [split_rings[0].coords]),  # 中间环带
    Polygon(outer_ring.coords, [split_rings[1].coords]),      # 外环区
    Polygon(split_rings[2].coords, [outer_ring.coords])       # 新增扩展环带
])

# 增强版数据导出
def export_ring_zones(zones: List[Polygon], names: List[str]):
    """导出四分区数据"""
    data = []
    for zone, name in zip(zones, names):
        valid_zone = GeometryValidator.validate_geometry(zone)
        wgs_zone = converter.convert_geometry(valid_zone, to_wgs84=True)
        
        data.append({
            '分区名称': name,
            'BD09_WKT': valid_zone.wkt,
            'WGS84_WKT': wgs_zone.wkt,
            '面积(m²)': round(valid_zone.area * 1e4, 2),  # 假设坐标单位为0.0001度≈10m
            '验证状态': '有效' if valid_zone.is_valid else '已修复'
        })
    
    with open('four_ring_zones.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['分区名称', 'BD09_WKT', 'WGS84_WKT', '面积(m²)', '验证状态'])
        writer.writeheader()
        writer.writerows(data)
    print("数据已导出至 four_ring_zones.csv")

# 可视化增强
def visualize_zones():
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_aspect('equal')
    
    # 绘制原始结构
    gpd.GeoSeries([Polygon(outer_ring)]).plot(ax=ax, color='grey', alpha=0.2, label='原始外环')
    if inner_ring:
        gpd.GeoSeries([Polygon(inner_ring)]).plot(ax=ax, color='red', alpha=0.3, label='原始内环')
    
    # 四分区样式配置
    zone_config = [
        {'color': '#1f77b4', 'label': '核心保护区'},     # 蓝色
        {'color': '#2ca02c', 'label': '缓冲区'},          # 绿色
        {'color': '#ff7f0e', 'label': '开发边界区'},      # 橙色
        {'color': '#9467bd', 'label': '扩展预留区', 'hatch': '///'} # 紫色带纹理
    ]
    
    # 分层绘制
    for idx, (zone, style) in enumerate(zip(zones, zone_config)):
        gdf = gpd.GeoSeries([zone]).plot(
            ax=ax,
            color=style['color'],
            alpha=0.4,
            edgecolor='black',
            hatch=style.get('hatch'),
            label=style['label']
        )
        
        # 添加面积标注
        centroid = zone.centroid
        ax.text(centroid.x, centroid.y, 
               f"{style['label']}\n{round(zone.area*1e4,2)}m²", 
               ha='center', va='center',
               fontsize=8, color='darkred')

    ax.legend(loc='upper right')
    plt.title("四环分层规划示意图（含扩展区）")
    plt.savefig('zones_visualization.png', dpi=300)
    plt.show()

# 执行流程调整
if __name__ == "__main__":
    export_ring_zones(zones, ['核心保护区', '缓冲区', '开发边界区', '扩展预留区'])
    visualize_zones()
