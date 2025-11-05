import os
import requests
from osgeo import gdal
from pyproj import Transformer

# -----------------------------
# CONFIGURATION
# -----------------------------
wms_url = "https://geoportal.muenchen.de/geoserver/gsm/Solarpotenzial_Globalstrahlung_p_02/wms"
layer_name = "Solarpotenzial_Globalstrahlung_p_02"
output_dir = "tiles"
merged_tif = "solar_munich.tif"

# Bounding box in EPSG:4326 (lon, lat)
bbox_4326 = [11.54, 48.11, 11.57, 48.14]

# Convert bbox to EPSG:25832 (UTM Zone 32N)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
minx, miny = transformer.transform(bbox_4326[0], bbox_4326[1])
maxx, maxy = transformer.transform(bbox_4326[2], bbox_4326[3])
bbox = [minx, miny, maxx, maxy]

# Tile grid
tiles_x = 8
tiles_y = 8
tile_width = 256
tile_height = 256

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 1. Download tiles
# -----------------------------
print("Downloading tiles...")
tile_files = []

dx = (bbox[2] - bbox[0]) / tiles_x
dy = (bbox[3] - bbox[1]) / tiles_y

for i in range(tiles_x):
    for j in range(tiles_y):
        xmin = bbox[0] + i * dx
        xmax = bbox[0] + (i + 1) * dx
        ymin = bbox[1] + j * dy
        ymax = bbox[1] + (j + 1) * dy

        url = (
            f"{wms_url}?service=WMS&version=1.1.1&request=GetMap"
            f"&layers={layer_name}&styles="
            f"&SRS=EPSG:25832"
            f"&bbox={xmin},{ymin},{xmax},{ymax}"
            f"&width={tile_width}&height={tile_height}"
            f"&format=image/png"
        )

        tile_png = os.path.join(output_dir, f"tile_{i}_{j}.png")
        r = requests.get(url)
        if r.status_code == 200 and r.content[:4] != b"<?xm":  # skip XML errors
            with open(tile_png, "wb") as f:
                f.write(r.content)
            tile_files.append((tile_png, xmin, xmax, ymin, ymax))
            print("✅", tile_png)
        else:
            print(f"❌ Failed tile {i},{j} HTTP {r.status_code}")

# -----------------------------
# 2. Flip tiles (180°) + convert to GeoTIFF
# -----------------------------
print("\nFlipping tiles and converting to GeoTIFF...")
tif_files = []

for tile_png, xmin, xmax, ymin, ymax in tile_files:
    tile_tif = tile_png.replace(".png", ".tif")

    # Flip vertically + horizontally by swapping bounds
    gdal.Translate(
        tile_tif,
        tile_png,
        outputSRS="EPSG:25832",
        outputBounds=[xmin, ymax, xmax, ymin],  # 180° flip
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE"]
    )
    tif_files.append(tile_tif)
    print("🗾", tile_tif)

# -----------------------------
# 3. Merge tiles into final GeoTIFF
# -----------------------------
print("\nMerging tiles into final GeoTIFF...")
gdal.Warp(
    merged_tif,
    tif_files,
    format="GTiff",
    creationOptions=["TILED=YES", "COMPRESS=DEFLATE"]
)
print(f"\n✅ DONE! Final merged raster: {merged_tif}")
print("Open in QGIS → Tiles should now align correctly.")
