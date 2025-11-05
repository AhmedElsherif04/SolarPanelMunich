import xml.etree.ElementTree as ET
import numpy as np

# --- 1️⃣ Load GML and namespaces ---
path = "/home/elsherif/Desktop/Thesis/Lod2/Neuberg/662_5400.gml"
tree = ET.parse(path)
root = tree.getroot()

namespaces = dict([
    node for _, node in ET.iterparse(path, events=['start-ns'])
])

# Building namespace
bldg_ns = None
for k, v in namespaces.items():
    if "building" in v.lower():
        bldg_ns = v
        break

# GML namespace
gml_ns = namespaces.get('gml', 'http://www.opengis.net/gml')

# --- 2️⃣ Loop over buildings ---
for i, building in enumerate(root.findall(f".//{{{bldg_ns}}}Building")):
    building_id = building.attrib.get("{http://www.opengis.net/gml}id")
    print(f"\n Building {i+1} (ID: {building_id}):")

    roof_points_all = []
    wall_points_all = []

    # Loop through boundedBy surfaces
    for bounded in building.findall(f".//{{{bldg_ns}}}boundedBy"):
        surface = list(bounded)[0] if list(bounded) else None
        if surface is None:
            continue

        surface_type = surface.tag.split("}")[-1]
        poslists = surface.findall(f".//{{{gml_ns}}}posList")

        for pl in poslists:
            coords = np.array(list(map(float, pl.text.strip().split()))).reshape(-1,3)

            if surface_type == "RoofSurface":
                roof_points_all.append(coords)
                # Compute tilt for this RoofSurface individually
                centroid = coords.mean(axis=0)
                _, _, vh = np.linalg.svd(coords - centroid)
                normal = vh[2,:]
                vertical = np.array([0,0,1])
                tilt_deg = np.degrees(np.arccos(np.clip(np.dot(normal, vertical)/np.linalg.norm(normal), -1,1)))
                print(f"  RoofSurface tilt: {tilt_deg:.2f}° (points: {len(coords)})")

            elif surface_type == "WallSurface":
                wall_points_all.append(coords)

    # --- 3️⃣ Compute roof centroid, footprint, height ---
    if roof_points_all:
        roof_points_all = np.vstack(roof_points_all)
        centroid = roof_points_all.mean(axis=0)
        x_center, y_center, z_center = centroid

        x_coords = roof_points_all[:,0]
        y_coords = roof_points_all[:,1]
        z_coords = roof_points_all[:,2]

        length = max(x_coords)-min(x_coords)
        width  = max(y_coords)-min(y_coords)
        roof_height_range = (min(z_coords), max(z_coords))

        print(f"  Roof centroid: ({x_center:.2f}, {y_center:.2f}, {z_center:.2f})")
        print(f"  Roof footprint (length x width): {length:.2f} x {width:.2f}")
        print(f"  Roof height range: {roof_height_range[0]:.2f} - {roof_height_range[1]:.2f} m")

    # --- 4️⃣ Wall height range ---
    if wall_points_all:
        wall_points_all = np.vstack(wall_points_all)
        z_coords_wall = wall_points_all[:,2]
        print(f"  Wall height range: {min(z_coords_wall):.2f} - {max(z_coords_wall):.2f} m")
