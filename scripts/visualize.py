
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import rasterio

# === Config ===
tiles_dir = "/Users/ariessunfeld/Documents/personal/UH/madeline-berger/data/planet/tifs/TILES_20220118_200909_30_2457_3B_Visual"
intersections_file = "intersections.txt"
coastline_file = "coastline_utm.gpkg"
coastline_layer = "coastline"

# === Load intersecting tile names ===
with open(intersections_file, "r") as f:
    intersecting_tiles = set(line.strip() for line in f if line.strip())

# === Build tile footprints ===
tile_geoms = []
tile_flags = []

for fname in os.listdir(tiles_dir):
    if not fname.endswith(".tif"):
        continue
    with rasterio.open(os.path.join(tiles_dir, fname)) as src:
        bounds = src.bounds
        geom = box(*bounds)
    tile_geoms.append(geom)
    tile_flags.append(fname in intersecting_tiles)

tiles_gdf = gpd.GeoDataFrame({"intersects": tile_flags}, geometry=tile_geoms, crs="EPSG:32604")

# === Load and simplify coastline ===
coastline = gpd.read_file(coastline_file, layer=coastline_layer)
coastline = coastline[coastline.is_valid & coastline.geometry.notnull()]
coastline["geometry"] = coastline.geometry.simplify(tolerance=100)

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 12))
tiles_gdf.plot(ax=ax, facecolor="none", edgecolor=tiles_gdf["intersects"].map({True: "red", False: "gray"}))
coastline.plot(ax=ax, color="blue", linewidth=0.5)
ax.set_title("Tile Footprints with Coastline Overlay\n(Red = Intersects Coastline)", fontsize=14)
plt.axis("equal")
plt.tight_layout()
plt.show()
