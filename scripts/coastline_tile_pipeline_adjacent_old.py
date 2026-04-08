import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
import geopandas as gpd
from rasterio.windows import Window
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input_tif = Path(__file__).parent.parent / 'data' / 'tifs' / '20220118_200909_30_2457_3B_Visual.tif'
coastline_shp = Path(__file__).parent.parent / 'data' / 'shapefiles' / 'coastline.shp'
output_root = Path(__file__).parent / "output"
tile_size = 512

tiles_dir = output_root / "tiles"
png_dir = output_root / "pngs"
output_csv = output_root / "intersections_extended.csv"
plot_output = output_root / "coastline_tiles_plot_extended.png"

tiles_dir.mkdir(parents=True, exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)

# === STEP 1: Tile the large TIFF ===
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
            with rasterio.open(tiles_dir / tile_filename, "w", **tile_profile) as dst:
                dst.write(tile_data)

    tif_bounds = box(*src.bounds)
    tif_crs = src.crs

# === STEP 2: Load and clip coastline ===
coastline = gpd.read_file(coastline_shp)
if coastline.crs != tif_crs:
    coastline = coastline.to_crs(tif_crs)

coastline = coastline[coastline.is_valid & coastline.geometry.notnull()]
# coastline_clip = coastline.clip(tif_bounds)
# coastline_outline = coastline_clip.copy()
# coastline_outline["geometry"] = coastline_outline.boundary
# coastline_union = coastline_outline.geometry.union_all()
# landmass_union = coastline_clip.geometry.union_all()
coastline_outline = coastline.copy()
coastline_outline["geometry"] = coastline_outline.boundary
coastline_union = coastline_outline.geometry.union_all()
landmass_union = coastline.geometry.union_all()

# === STEP 3: Build tile GeoDataFrame ===
tile_records = []
for tif_file in os.listdir(tiles_dir):
    if not tif_file.endswith(".tif"):
        continue
    with rasterio.open(tiles_dir / tif_file) as src:
        bounds = src.bounds
        tile_records.append({
            "tile": tif_file,
            "geometry": box(*bounds),
            "extent": (bounds.left, bounds.right, bounds.bottom, bounds.top)
        })

tiles_gdf = gpd.GeoDataFrame(tile_records, crs=tif_crs)

# === STEP 4: Identify intersecting tiles (coastline + content check) ===
intersecting_tiles = []
for idx, row in tiles_gdf.iterrows():
    tile_path = tiles_dir / row["tile"]
    with rasterio.open(tile_path) as src:
        if not row["geometry"].intersects(coastline_union):
            continue
        img = src.read([1, 2, 3])
        if np.all(img == 0) or np.all(np.isnan(img)):
            continue
        mask = src.read_masks(1)
        if mask.mean() < 10:
            continue
    intersecting_tiles.append(row["tile"])

coast_tiles = tiles_gdf[tiles_gdf["tile"].isin(intersecting_tiles)]

# === STEP 5: Find adjacent tiles (touching) and filter out landlocked ones ===
adjacent_tiles = tiles_gdf[
    ~tiles_gdf["tile"].isin(coast_tiles["tile"]) &
    tiles_gdf["geometry"].apply(lambda g: any(g.touches(cg) for cg in coast_tiles.geometry))
]
adjacent_tiles = adjacent_tiles[
    ~adjacent_tiles["geometry"].apply(lambda g: g.within(landmass_union))
]

# === STEP 6: Label tile categories ===
coast_tiles = coast_tiles.copy()
adjacent_tiles = adjacent_tiles.copy()
coast_tiles["category"] = "intersecting"
adjacent_tiles["category"] = "adjacent"
other_tiles = tiles_gdf[
    ~tiles_gdf["tile"].isin(coast_tiles["tile"]) &
    ~tiles_gdf["tile"].isin(adjacent_tiles["tile"])
].copy()
other_tiles["category"] = "other"

all_tiles = pd.concat([coast_tiles, adjacent_tiles, other_tiles])
all_tiles.to_csv(output_csv, index=False)

# === STEP 7: Generate PNGs for intersecting + adjacent tiles ===
for _, row in all_tiles[all_tiles["category"].isin(["intersecting", "adjacent"])].iterrows():
    tif_path = tiles_dir / row["tile"]
    png_path = png_dir / row["tile"].replace(".tif", ".png")
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0)).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(png_path)

# === STEP 8: Plot filled tiles with outlines ===
fig, ax = plt.subplots(figsize=(12, 12))

# TIFF extent
gpd.GeoSeries([tif_bounds], crs=tif_crs).boundary.plot(ax=ax, edgecolor="black", linewidth=1, label="TIFF extent")

# Coastline
# coastline_outline.plot(ax=ax, color="blue", linewidth=1.5, label="Coastline")
coastline_plot = coastline_outline.clip(tif_bounds)
coastline_plot.plot(ax=ax, color="blue", linewidth=1.5, label='Coastline')

# Category outlines
colors = {"intersecting": "red", "adjacent": "orange", "other": "black"}
for cat, group in all_tiles.groupby("category"):
    group.boundary.plot(ax=ax, edgecolor=colors[cat], linewidth=1, label=f"{cat.title()} Tiles")

# PNG fill
for _, row in all_tiles[all_tiles["category"].isin(["intersecting", "adjacent"])].iterrows():
    png_path = png_dir / row["tile"].replace(".tif", ".png")
    if png_path.exists():
        img = Image.open(png_path)
        ax.imshow(img, extent=row["extent"], origin="upper")

ax.set_title("Coastline-Intersecting and Adjacent PNG Tiles")
ax.axis("equal")
ax.axis("off")
ax.legend()
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
print(f"Saved extended plot to: {plot_output}")
