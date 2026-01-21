#!/usr/bin/env python3
"""
Add district name and year to tiles summary and export as CSV.
Uses centroid coordinates from each tile.
Falls back to nearest district if point is not contained in any district.

Requires:
    pip install shapely pyproj pandas
"""
import json
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape, Point
from collections import Counter

def load_geojson(filepath):
    """Load GeoJSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_districts(districts_file):
    """
    Load district boundaries and convert to shapely polygons.
    Assumes districts are in UTM coordinates (EPSG:25832).
    
    Returns:
        list: List of dicts with 'name', 'sb_nummer', and 'polygon'
    """
    print(f"📍 Loading districts from {districts_file}...")
    districts_data = load_geojson(districts_file)
    
    district_polygons = []
    for district in districts_data['features']:
        poly = shape(district['geometry'])
        district_polygons.append({
            'name': district['properties'].get('name', 'Unknown'),
            'sb_nummer': district['properties'].get('sb_nummer', 'N/A'),
            'polygon': poly
        })
    
    print(f"✅ Loaded {len(district_polygons)} districts")
    return district_polygons

def find_district_for_point(lon, lat, district_polygons, transformer):
    """
    Find which district contains the given WGS84 point.
    If not contained, find nearest district.
    
    Returns:
        tuple: (name, sb_nummer, distance_meters, matched_type)
               matched_type is 'exact' or 'nearest'
    """
    utm_x, utm_y = transformer.transform(lon, lat)
    utm_point = Point(utm_x, utm_y)
    
    # First try exact match
    for district in district_polygons:
        if district['polygon'].contains(utm_point):
            return district['name'], district['sb_nummer'], 0.0, 'exact'
    
    # If no exact match, find nearest
    min_distance = float('inf')
    nearest_district = None
    
    for district in district_polygons:
        distance = district['polygon'].distance(utm_point)
        if distance < min_distance:
            min_distance = distance
            nearest_district = district
    
    if nearest_district:
        return (
            nearest_district['name'],
            nearest_district['sb_nummer'],
            min_distance,
            'nearest'
        )
    
    return None, None, None, None

def add_district_and_year_to_tiles(
    tiles_json_path,
    districts_json_path,
    output_csv_path,
    year
):
    """
    Add district and year to tiles summary (CSV output).
    Uses centroid_lon and centroid_lat from each tile.
    Falls back to nearest district if not contained.
    """
    print("=" * 70)
    print("ADDING DISTRICT AND YEAR TO TILES (CSV OUTPUT)")
    print("=" * 70)
    
    # Load tiles summary
    print(f"\n📂 Loading tiles from {tiles_json_path}...")
    with open(tiles_json_path, 'r') as f:
        tiles = json.load(f)
    print(f"✅ Loaded {len(tiles)} tiles")
    
    # Load districts
    district_polygons = load_districts(districts_json_path)
    
    # Create transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    
    enhanced_tiles = []
    exact_matched = 0
    nearest_matched = 0
    no_coords = 0
    nearest_matches = []  # Store info about nearest matches
    
    for i, tile in enumerate(tiles):
        tile_name = tile.get('tile', f'tile_{i}')
        centroid_lon = tile.get('tile_centroid_lon')
        centroid_lat = tile.get('tile_centroid_lat')
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tiles)} tiles...")
        
        district_name, district_number = "Unknown", "N/A"
        match_distance = None
        match_type = None
        
        if centroid_lon is not None and centroid_lat is not None:
            name, num, distance, mtype = find_district_for_point(
                centroid_lon, centroid_lat, district_polygons, transformer
            )
            
            if name:
                district_name, district_number = name, num
                match_distance = distance
                match_type = mtype
                
                if mtype == 'exact':
                    exact_matched += 1
                elif mtype == 'nearest':
                    nearest_matched += 1
                    nearest_matches.append({
                        'tile': tile_name,
                        'district': district_name,
                        'distance_m': round(distance, 2),
                        'lon': centroid_lon,
                        'lat': centroid_lat
                    })
        else:
            no_coords += 1
            print(f"⚠️  Tile '{tile_name}' missing centroid coordinates")
        
        # Add year and district info
        tile['district_name'] = district_name
        tile['district_number'] = district_number
        tile['year'] = year
        tile['match_type'] = match_type if match_type else 'no_coords'
        tile['distance_to_district_m'] = match_distance
        
        enhanced_tiles.append(tile)
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_tiles)
    
    # Save to CSV
    print(f"\n💾 Saving to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved {len(df)} rows → {output_csv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    district_counts = Counter(df['district_name'])
    for d, count in district_counts.most_common():
        print(f"  {d}: {count} tiles")
    
    print(f"\n✅ Exact matches (point inside district): {exact_matched} tiles")
    print(f"🔍 Nearest matches (outside all districts): {nearest_matched} tiles")
    print(f"⚠️  No coordinates: {no_coords} tiles")
    
    # Report nearest matches with distances
    if nearest_matches:
        print(f"\n" + "=" * 70)
        print(f"NEAREST MATCHES DETAILS ({len(nearest_matches)} tiles)")
        print("=" * 70)
        
        # Sort by distance
        nearest_matches.sort(key=lambda x: x['distance_m'])
        
        for match in nearest_matches:
            print(f"  📍 {match['tile']}")
            print(f"     → District: {match['district']}")
            print(f"     → Distance: {match['distance_m']} meters")
            print(f"     → Coords: ({match['lon']:.6f}, {match['lat']:.6f})")
            print()
        
        # Statistics
        distances = [m['distance_m'] for m in nearest_matches]
        print(f"  Distance statistics:")
        print(f"    Min: {min(distances):.2f} m")
        print(f"    Max: {max(distances):.2f} m")
        print(f"    Avg: {sum(distances)/len(distances):.2f} m")
    
    print(f"\n🎯 Year added: {year}")
    print(f"✅ Done! CSV saved to: {output_csv_path}")
    
    return df

def main():
    print("=" * 70)
    print("MUNICH TILES - ADD DISTRICT & YEAR (CSV OUTPUT)")
    print("=" * 70)
    
    # Ask for year
    while True:
        year_input = input("\n📅 Enter the year for this data: ").strip()
        try:
            year = int(year_input)
            if 2000 <= year <= 2100:
                break
            else:
                print("⚠️  Please enter a valid year between 2000 and 2100")
        except ValueError:
            print("⚠️  Please enter a valid year (e.g., 2024)")
    
    tiles_json_path = input("\n📂 Path to tiles JSON [GeoJsontoZip/summarytest.json]: ").strip() or \
        "/home/elsherif/Desktop/Thesis/ViewPython/data/Rooftop_Data/unclean/summary2018.json"
    
    districts_json_path = input("📂 Path to districts GeoJSON [GeoJsontoZip/areasMunich.json]: ").strip() or \
        "GeoJsontoZip/areasMunich.json"
    
    output_csv_path = input("📂 Output CSV file [GeoJsontoZip/tiles_with_district_year.csv]: ").strip() or \
        "/home/elsherif/Desktop/Thesis/ViewPython/data/Rooftop_Data/clean/Rooftop_with_district_2018.csv"
    
    add_district_and_year_to_tiles(
        tiles_json_path=tiles_json_path,
        districts_json_path=districts_json_path,
        output_csv_path=output_csv_path,
        year=year
    )

if __name__ == "__main__":
    main()