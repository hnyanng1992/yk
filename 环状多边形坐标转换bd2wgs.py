from typing import List, Tuple, Union
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint, LineString
from shapely.ops import nearest_points, transform
from pyproj import Transformer
import math

class CoordinateConverter:
    """
    高级坐标系转换工具类
    支持WGS84、GCJ02、BD09坐标系之间的相互转换
    支持各种几何类型（Point、LineString、Polygon、MultiPolygon）的批量转换
    """
    
    def __init__(self):
        # 初始化转换器（使用中国官方参数）
        self.gcj02_to_wgs84 = Transformer.from_crs("EPSG:4490", "EPSG:4326", always_xy=True)
        self.wgs84_to_gcj02 = Transformer.from_crs("EPSG:4326", "EPSG:4490", always_xy=True)
    
    @property
    def x_pi(self):
        """BD09坐标系专用参数"""
        return math.pi * 3000.0 / 180.0
    
    # ================== 单点坐标转换 ==================
    def bd09_to_gcj02(self, lon: float, lat: float) -> Tuple[float, float]:
        """BD09转GCJ02坐标系"""
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x*x + y*y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        return z * math.cos(theta), z * math.sin(theta)
    
    def gcj02_to_bd09(self, lon: float, lat: float) -> Tuple[float, float]:
        """GCJ02转BD09坐标系"""
        z = math.sqrt(lon*lon + lat*lat) + 0.00002 * math.sin(lat * self.x_pi)
        theta = math.atan2(lat, lon) + 0.000003 * math.cos(lon * self.x_pi)
        return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006
    
    def bd09_to_wgs84(self, lon: float, lat: float) -> Tuple[float, float]:
        """BD09转WGS84坐标系（通过GCJ02中转）"""
        gcj_lon, gcj_lat = self.bd09_to_gcj02(lon, lat)
        return self.gcj02_to_wgs84.transform(gcj_lon, gcj_lat)
    
    def wgs84_to_bd09(self, lon: float, lat: float) -> Tuple[float, float]:
        """WGS84转BD09坐标系（通过GCJ02中转）"""
        gcj_lon, gcj_lat = self.wgs84_to_gcj02.transform(lon, lat)
        return self.gcj02_to_bd09(gcj_lon, gcj_lat)
    
    # ================== 几何对象转换 ==================
    def convert_geometry(self, geom: Union[Point, LineString, Polygon, MultiPolygon], 
                        to_wgs84: bool = True) -> Union[Point, LineString, Polygon, MultiPolygon]:
        """
        通用几何对象坐标系转换
        :param geom: 输入几何对象（支持Point/LineString/Polygon/MultiPolygon）
        :param to_wgs84: True表示转WGS84，False表示转BD09
        :return: 转换后的几何对象
        """
        if geom is None or geom.is_empty:
            return geom
            
        if isinstance(geom, Point):
            return self._convert_point(geom, to_wgs84)
        elif isinstance(geom, LineString):
            return self._convert_linestring(geom, to_wgs84)
        elif isinstance(geom, Polygon):
            return self._convert_polygon(geom, to_wgs84)
        elif isinstance(geom, MultiPolygon):
            return self._convert_multipolygon(geom, to_wgs84)
        return geom
    
    def _convert_point(self, point: Point, to_wgs84: bool) -> Point:
        """点对象转换"""
        coords = list(point.coords)[0]
        if to_wgs84:
            new_coords = self.bd09_to_wgs84(*coords)
        else:
            new_coords = self.wgs84_to_bd09(*coords)
        return Point(new_coords)
    
    def _convert_linestring(self, line: LineString, to_wgs84: bool) -> LineString:
        """线对象转换"""
        coords = list(line.coords)
        if to_wgs84:
            new_coords = [self.bd09_to_wgs84(*c) for c in coords]
        else:
            new_coords = [self.wgs84_to_bd09(*c) for c in coords]
        return LineString(new_coords)
    
    def _convert_polygon(self, polygon: Polygon, to_wgs84: bool) -> Polygon:
        """多边形对象转换"""
        # 转换外环
        exterior = list(polygon.exterior.coords)
        if to_wgs84:
            new_exterior = [self.bd09_to_wgs84(*c) for c in exterior]
        else:
            new_exterior = [self.wgs84_to_bd09(*c) for c in exterior]
        
        # 转换所有内环
        new_interiors = []
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            if to_wgs84:
                new_interior = [self.bd09_to_wgs84(*c) for c in interior_coords]
            else:
                new_interior = [self.wgs84_to_bd09(*c) for c in interior_coords]
            new_interiors.append(new_interior)
        
        return Polygon(new_exterior, new_interiors)
    
    def _convert_multipolygon(self, multipolygon: MultiPolygon, to_wgs84: bool) -> MultiPolygon:
        """多多边形对象转换"""
        return MultiPolygon([self._convert_polygon(poly, to_wgs84) for poly in multipolygon.geoms])
    
    # ================== 批量转换方法 ==================
    def convert_coordinates(self, coordinates: List[Tuple[float, float]], 
                          to_wgs84: bool = True) -> List[Tuple[float, float]]:
        """
        批量转换坐标列表
        :param coordinates: 坐标列表 [(lon, lat), ...]
        :param to_wgs84: True表示转WGS84，False表示转BD09
        :return: 转换后的坐标列表
        """
        if to_wgs84:
            return [self.bd09_to_wgs84(*c) for c in coordinates]
        else:
            return [self.wgs84_to_bd09(*c) for c in coordinates]
    
    def convert_wkt(self, wkt_str: str, to_wgs84: bool = True) -> str:
        """
        WKT字符串坐标系转换
        :param wkt_str: WKT格式字符串
        :param to_wgs84: True表示转WGS84，False表示转BD09
        :return: 转换后的WKT字符串
        """
        from shapely import wkt
        geom = wkt.loads(wkt_str)
        converted = self.convert_geometry(geom, to_wgs84)
        return converted.wkt

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 初始化转换器
    converter = CoordinateConverter()
    
    # 示例MULTIPOLYGON或polygon WKT
    multipolygon_wkt = """"""
    
    # 方法1：直接转换WKT字符串
    converted_wkt = converter.convert_wkt(multipolygon_wkt, to_wgs84=True)
    print("转换后的WKT:", converted_wkt)
    
    # 方法2：加载为几何对象后转换
    from shapely import wkt
    multipolygon = wkt.loads(multipolygon_wkt)
    converted_geom = converter.convert_geometry(multipolygon, to_wgs84=True)
    print("转换后的几何对象类型:", converted_geom.geom_type)
    
    # 方法3：批量转换坐标列表
    # coords = [(116.404, 39.915), (116.405, 39.916)]  # 示例坐标
    # converted_coords = converter.convert_coordinates(coords, to_wgs84=True)
    # print("转换后的坐标:", converted_coords)
