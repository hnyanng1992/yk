地图环状数据处理方法（经过多轮deepseek交互得到可靠代码）
==

## 几何数据展示工具（Vercel 静态页）

在线地址：[https://maptool-pi.vercel.app/](https://maptool-pi.vercel.app/)

### 功能

- 点 / 线 / 多边形 WKT、坐标串、GeoJSON 可视化
- **双底图切换**：高德（GCJ02）/ 百度（BD09）
- WGS84/CGCS2000 → 展示坐标系自动转换
- 卫星影像叠加、戳点拾取、POI 标注、多边形自相交检测

### 百度底图配置

1. 在 [百度地图开放平台](https://lbsyun.baidu.com/) 申请浏览器端 AK
2. 将 `index.html` 中 `CONFIG.BAIDU_AK` 替换为你的 AK
3. 重新部署到 Vercel

### 本地预览

```bash
cd vercel地图工具
python3 -m http.server 8080
# 打开 http://localhost:8080
```

---

1、环路线转面，包括孔洞和不含孔洞处理

2、坐标转换，bd09和wgs转换。现已支持百度底图直接校验 BD09 数据

3、环状多边形，使用中心点射线法，三等分，进行空间分析或者可视化

4、环状多边形的合并

5、环状多边形的东西南北8方位切分
