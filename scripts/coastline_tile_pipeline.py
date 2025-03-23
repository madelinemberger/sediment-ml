
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
import geopandas as gpd
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input_tif = Path(__file__).parent.parent / 'data' / 'tifs' / '20220118_200909_30_2457_3B_Visual.tif'
coastline_shp = Path(__file__).parent.parent / 'data' / 'shapefiles' / 'coastline.shp' / 'coastline.shp'
output_root = "output"
tile_size = 512

tiles_dir = os.path.join(output_root, "tiles")
png_dir = os.path.join(output_root, "pngs")
output_csv = os.path.join(output_root, "intersections.csv")
plot_output = os.path.join(output_root, "coastline_tiles_plot.png")

os.makedirs(tiles_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

with rasterio.open(input_tif) as src:
    n_cols, n_rows = src.width, src.height
    profile = src.profile

    for i in range(0, n_rows, tile_size):
        for j in range(0, n_cols, tile_size):
            row, col = i // tile_size, j // tile_size
            width = min(tile_size, n_cols - j)
            height = min(tile_size, n_rows - i)
            window = Window(j, i, width, height)
            transform = src.window_transform(window)

            tile_data = src.read(window=window)
            tile_profile = profile.copy()
            tile_profile.update({
                "height": height,
                "width": width,
                "transform": transform
            })

            tile_filename = f"tile_{row}_{col}.tif"
            with rasterio.open(os.path.join(tiles_dir, tile_filename), "w", **tile_profile) as dst:
                dst.write(tile_data)

    tif_bounds = box(*src.bounds)
    tif_crs = src.crs


coastline = gpd.read_file(coastline_shp)
if coastline.crs != tif_crs:
    coastline = coastline.to_crs(tif_crs)

coastline = coastline[coastline.is_valid & coastline.geometry.notnull()]
coastline_clip = coastline.clip(tif_bounds)
coastline_outline = coastline_clip.copy()
coastline_outline["geometry"] = coastline_outline.boundary
coastline_union = coastline_outline.geometry.union_all()

intersecting_tiles = []

for fname in os.listdir(tiles_dir):
    if not fname.endswith(".tif"):
        continue
    path = os.path.join(tiles_dir, fname)

    with rasterio.open(path) as src:
        bounds = box(*src.bounds)
        if not bounds.intersects(coastline_union):
            continue

        img = src.read([1, 2, 3])
        if np.all(img == 0) or np.all(np.isnan(img)):
            continue

        mask = src.read_masks(1)
        if mask.mean() < 10:
            continue

        intersecting_tiles.append(fname)

pd.DataFrame({"tile": intersecting_tiles}).to_csv(output_csv, index=False)

for fname in intersecting_tiles:
    tif_path = os.path.join(tiles_dir, fname)
    png_path = os.path.join(png_dir, fname.replace(".tif", ".png"))

    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0)).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(png_path)

tile_records = []
for fname in os.listdir(png_dir):
    if not fname.endswith(".png"):
        continue
    tif_equiv = fname.replace(".png", ".tif")
    tif_path = os.path.join(tiles_dir, tif_equiv)
    if not os.path.exists(tif_path):
        continue
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tile_records.append({
            "tile": fname,
            "geometry": box(*bounds),
            "extent": (bounds.left, bounds.right, bounds.bottom, bounds.top)
        })

tiles_gdf = gpd.GeoDataFrame(tile_records, crs=tif_crs)

fig, ax = plt.subplots(figsize=(12, 12))

# TIFF boundary
gpd.GeoSeries([tif_bounds], crs=tif_crs).boundary.plot(ax=ax, edgecolor="black", linewidth=1, label="TIFF extent")

# Coastline
coastline_outline.plot(ax=ax, color="blue", linewidth=1.5, label="Coastline")

# Empty tile outlines (non-intersecting)
all_tile_boxes = [box(*rasterio.open(os.path.join(tiles_dir, f.replace(".png", ".tif"))).bounds)
                  for f in os.listdir(png_dir)]
full_tiles_gdf = gpd.GeoDataFrame(geometry=all_tile_boxes, crs=tif_crs)
full_tiles_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

# PNG tiles and red outlines
for idx, row in tiles_gdf.iterrows():
    img = Image.open(os.path.join(png_dir, row["tile"]))
    ax.imshow(img, extent=row["extent"], origin="upper")
    gpd.GeoSeries([row["geometry"]], crs=tif_crs).boundary.plot(ax=ax, edgecolor="red", linewidth=1)

ax.set_title("Coastline-Intersecting PNG Tiles")
ax.axis("equal")
ax.axis("off")
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
print(f"Saved plot to: {plot_output}")


