from shapely.wkt import loads
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def extract_coordinates(polygon):
    coordinates = []
    if polygon.geom_type == 'Polygon':
        exterior = list(polygon.exterior.coords)
        coordinates.append(exterior)
        for interior in polygon.interiors:
            coordinates.append(list(interior.coords))
    elif polygon.geom_type == 'MultiPolygon':
        for p in polygon.geoms:
            coordinates.extend(extract_coordinates(p))
    return coordinates

def round_coords(coords, precision=6):
    return [(round(x, precision), round(y, precision)) for x, y in coords]

def merge_rings(wkt_str: str, save_path: str = "merged_optimized.wkt"):
    multi_polygon = loads(wkt_str)
    all_rings = extract_coordinates(multi_polygon)
    
    # 动态计算阈值
    areas = [Polygon(ring).area for ring in all_rings]
    max_area = max(areas)
    threshold = max_area * 0.01

    # 分离内外环
    exterior_rings = [ring for ring in all_rings if Polygon(ring).area > threshold]
    interior_rings = [ring for ring in all_rings if Polygon(ring).area <= threshold]

    # 容错处理
    if not exterior_rings:
        exterior_rings = [max(all_rings, key=lambda r: Polygon(r).area)]
    if not interior_rings:
        interior_rings = [min(all_rings, key=lambda r: Polygon(r).area)]

    # 处理坐标精度（新增部分）
    largest_exterior = round_coords(max(exterior_rings, key=lambda r: Polygon(r).area))
    smallest_interior = round_coords(min(interior_rings, key=lambda r: Polygon(r).area))

    # 构建多边形
    merged_polygon = Polygon(largest_exterior, [smallest_interior]).buffer(0)
    
    # 保存并输出结果
    with open(save_path, 'w') as f:
        f.write(merged_polygon.wkt)
    print(f"结果已保存至: {save_path} (六位小数精度)")

    # 可视化
    gpd.GeoSeries([merged_polygon]).plot(edgecolor='red', facecolor='none')
    plt.title("合并结果（动态阈值）")
    plt.show()

if __name__ == "__main__":
    input_wkt = """"""  # 实际数据已省略
    merge_rings(input_wkt)
