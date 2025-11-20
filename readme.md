# Munich PV Potential Dataset

This repository provides a comprehensive dataset for **forecasting future photovoltaic (PV) potential** in Munich. It combines geospatial, solar, demographic, and rooftop-level data.

---

## Data Sources

### 1. Munich Open Data Portal
[Geoportal München Open Data](https://geoportal.muenchen.de/portal/opendata/#LayerInfoDataDownload)  
Layers used:

1. **Stadtbezirke der Landeshauptstadt München** – District boundaries in Munich  
2. **Solarpotenzial_Globalstrahlung** – Solar potential (*obtained via `data/Woms Data/downloadwoms.py`*)  
3. **Digitaler Flächennutzungsplan der Landeshauptstadt München** – Digital land use plan (*obtained via `data/Woms Data/downloadwoms.py`*)  

### 2. Munich Statistical Office
[München Indikatorenatlas](https://mstatistik.muenchen.de/indikatorenatlas/export/export.php)  
District-level demographic indicators:

- Stand-Alone Houses  
- Average Population Age  
- Overall Population  
- Young Population  
- Senior Population  
- Unemployment Rate  

---

## Dataset Description

The dataset is structured as **tiles**, each representing a **1000 × 1000 m area** in Munich. The dataset covers the years:
2003, 2006, 2009, 2012, 2015, 2018, 2020, 2022, 2024


### Final Dataset (`data/Train Data/rooftop.csv`)

| Variable | Description |
|----------|-------------|
| `tile` | Unique tile identifier (e.g., `tile_r0_c0`) |
| `total_rooftops` | Total number of rooftops in the tile |
| `rooftops_without_solar` | Number of rooftops without solar panels |
| `square_meters_with_solar_m2` | Total rooftop area (in m²) that contain solar panels |
| `panel_area_m2` | Actual area (in m²) of PV panels detected |
| `district_number` | District number in Munich |
| `year` | Year of observation |
| `Unemployment_Rate` | District-level unemployment rate (%) |
| `Average_Age` | Average population age in the district (years) |
| `Elderly_Population` | Number of residents aged 65+ |
| `Young_Population` | Number of residents aged 0–18 |
| `Total_Population` | Total number of residents in the district |
| `Number_of_Houses` | Total number of houses in the district |

> **Note:** `Unemployment_Rate` and `Average_Age` use comma as decimal (e.g., `"3,8"` → 3.8%).

---

### Previous Version (`data/Rooftop_Data/clean`)

The previous version of the dataset included additional geospatial information derived from **house segmentation** and **solar panel segmentation models**:

| Variable | Description |
|----------|-------------|
| `square_meters_without_solar_m2` | Rooftop area without solar panels |
| `tile_centroid_lat` | Latitude of tile centroid |
| `tile_centroid_lon` | Longitude of tile centroid |
| `district_name` | Name of the district |
| `match_type` | Method for associating tiles with districts (`exact` or `nearest`) |
| `distance_to_district_m` | Distance from tile centroid to district center (meters) |

---

## Notes

- Data from solar potential and land use layers cannot be downloaded directly; use `downloadwoms.py`.  
- Each tile represents a **1000 × 1000 m area**.  
- The dataset allows analysis of rooftop-level PV adoption, solar potential, and demographic effects on PV deployment.

---

## Citation

If you use this dataset in your research, please reference this repository.

