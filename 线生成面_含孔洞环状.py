import csv
import math
from typing import List, Tuple

from shapely import wkt, geometry, validation, ops
from pyproj import Transformer


class CoordinateConverter:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:4490", "EPSG:4326", always_xy=True)  # GCJ02/WGS84 fallback

    def bd09_to_wgs84(self, lon: float, lat: float) -> Tuple[float, float]:
        x_pi = math.pi * 3000.0 / 180.0
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        wgs_lon, wgs_lat = self.transformer.transform(gcj_lon, gcj_lat)
        return round(wgs_lon, 6), round(wgs_lat, 6)

    def convert_batch(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.bd09_to_wgs84(lon, lat) for lon, lat in points]


class GeometryValidator:
    @staticmethod
    def process_ring(wkt_string: str) -> geometry.LineString:
        geom = wkt.loads(wkt_string)
        geom = validation.make_valid(geom)

        if geom.geom_type == 'MultiLineString':
            geom = ops.linemerge(geom)

        coords = list(geom.coords)

        if len(coords) < 4:
            raise ValueError("Ring must have at least 4 coordinates (including closing point)")

        # 判断是否闭环（考虑浮点误差）
        if not GeometryValidator._is_closed(coords):
            coords.append(coords[0])

        return geometry.LineString(coords)

    @staticmethod
    def _is_closed(coords: List[Tuple[float, float]], tol=1e-6) -> bool:
        return (abs(coords[0][0] - coords[-1][0]) < tol and abs(coords[0][1] - coords[-1][1]) < tol)


class WKTGenerator:
    @staticmethod
    def create_polygon(exterior: List[Tuple[float, float]], interiors: List[List[Tuple[float, float]]] = []) -> geometry.Polygon:
        return geometry.Polygon(exterior, interiors)
    
    @staticmethod
    def round_coords(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(round(x, 6), round(y, 6)) for x, y in coords]

    @staticmethod
    def create_polygon(exterior: List[Tuple[float, float]], interiors: List[List[Tuple[float, float]]] = []) -> geometry.Polygon:
        ext = WKTGenerator.round_coords(exterior)
        ints = [WKTGenerator.round_coords(ring) for ring in interiors]
        return geometry.Polygon(ext, ints)



class CSVExporter:
    @staticmethod
    def export(data: List[dict], filename: str = 'polygon_data.csv'):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['polygon_bd09_wkt', 'polygon_wgs84_wkt'])
            writer.writeheader()
            for row in data:
                writer.writerow(row)


def main():
    # 示例 WKT：一个外环 + 一个内环，linestring 或者 multilinestring
    # 这里可以替换为实际的 WKT 字符串
    inner_wkt = ""
    
    outer_wkt = "" 
      
    # 处理环
    outer_ring = GeometryValidator.process_ring(outer_wkt)
    inner_ring = GeometryValidator.process_ring(inner_wkt)

    # 提取 BD09 坐标
    bd09_outer = list(outer_ring.coords)
    bd09_inner = list(inner_ring.coords)

    # 转换为 WGS84
    converter = CoordinateConverter()
    wgs84_outer = converter.convert_batch(bd09_outer)
    wgs84_inner = converter.convert_batch(bd09_inner)

    # 构造多边形对象
    polygon_bd09 = WKTGenerator.create_polygon(bd09_outer, [bd09_inner])
    polygon_wgs84 = WKTGenerator.create_polygon(wgs84_outer, [wgs84_inner])

    # 导出为 CSV
    data = [{
        'polygon_bd09_wkt': polygon_bd09.wkt,
        'polygon_wgs84_wkt': polygon_wgs84.wkt
    }]
    CSVExporter.export(data)
    print("Exported polygon_data.csv")


if __name__ == "__main__":
    main()

   
