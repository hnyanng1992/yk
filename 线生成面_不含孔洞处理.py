from shapely.geometry import Polygon, MultiLineString
from shapely import wkt, ops, validation
import math
import numpy as np
from pyproj import Transformer

# 预初始化坐标转换器
GCJ02_TO_WGS84 = Transformer.from_crs("EPSG:4490", "EPSG:4326")

def bd09_to_wgs84(lon, lat):
    """BD09坐标系转WGS84坐标系（优化版）"""
    # BD09转GCJ02
    x_pi = math.pi * 3000.0 / 180.0
    x = lon - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gcj_lon, gcj_lat = z * math.cos(theta), z * math.sin(theta)
    # GCJ02转WGS84
    return GCJ02_TO_WGS84.transform(gcj_lon, gcj_lat)

def convert_coords(coords, transformer):
    """批量坐标转换（处理嵌套环结构）"""
    return [np.array([transformer(x, y) for x, y in ring]) for ring in coords]

def ensure_valid_polygon(poly):
    """确保多边形有效性（带缓冲修复）"""
    if not poly.is_valid:
        poly = poly.buffer(0)
    return validation.make_valid(poly)

def process_rings(poly):
    """提取多边形环结构（外环+内环）"""
    return [np.array(poly.exterior.coords)] + [np.array(ring.coords) for ring in poly.interiors]

def convert_linestring_to_dual_polygons(wkt_input):
    """双坐标系多边形生成核心逻辑"""
    # 原始BD09多边形处理
    geom = wkt.loads(wkt_input)
    if isinstance(geom, MultiLineString):
        geom = ops.linemerge(geom)
    
    if len(geom.coords) < 3:
        raise ValueError("至少需要3个点构成多边形")
    
    # 坐标处理（BD09）
    bd_coords = np.round(np.array(geom.coords), 6)
    _, idx = np.unique(bd_coords, axis=0, return_index=True)
    bd_coords = bd_coords[np.sort(idx)]
    if not np.array_equal(bd_coords[0], bd_coords[-1]):
        bd_coords = np.vstack([bd_coords, bd_coords[0]])
    
    # 创建BD09多边形
    bd_poly = ensure_valid_polygon(Polygon(bd_coords))
    
    # 转换WGS84坐标系
    rings = process_rings(bd_poly)
    wgs_rings = []
    for ring in rings:
        wgs_ring = np.array([bd09_to_wgs84(x, y) for x, y in ring])
        wgs_rings.append(wgs_round(wgs_ring))
    
    # 创建WGS84多边形
    wgs_exterior = wgs_rings[0]
    wgs_interiors = wgs_rings[1:] if len(wgs_rings) > 1 else []
    wgs_poly = ensure_valid_polygon(Polygon(wgs_exterior, wgs_interiors))
    
    return bd_poly.wkt, wgs_poly.wkt

def wgs_round(coords, precision=6):
    """WGS84坐标精确舍入"""
    return np.round(coords, decimals=precision)

def main():
    """输入linestring，输出双坐标系多边形"""
    input_linestring =""
    
    try:
        bd_wkt, wgs_wkt = convert_linestring_to_dual_polygons(input_linestring)
        print("BD09多边形:\n", bd_wkt)
        print("\nWGS84多边形:\n", wgs_wkt)
    except Exception as e:
        print(f"转换失败：{str(e)}")

if __name__ == "__main__":
    main()

 
