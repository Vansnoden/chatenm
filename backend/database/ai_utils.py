import json
import shutil
from typing import Annotated
import uuid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
# from IPython.display import Image, display
import getpass
from langchain_tavily import TavilySearch
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import requests
import zipfile
import tempfile
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import requests
import pandas as pd
import geopandas as gpd
import pandas as pd
import elapid as ela
import os, zipfile, requests, io
from typing import Dict, Any
import rasterio.mask
from shapely.geometry import mapping
from sklearn.metrics import roc_auc_score
import os
import glob
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import geopandas as gpd
import os
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from typing import Dict, Any, Optional
import geodatasets
import pystac_client
import planetary_computer
import rioxarray
from typing import Optional
import seaborn as sns
from shapely.geometry import Point
from typing import Dict, Any, Optional
import ee
import geemap
import os
import time
from dotenv import load_dotenv
import random

load_dotenv() 


# wordclim base url:
GADM_JSON_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{level}.json"
# gbfif base url:
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_API = "https://api.gbif.org/v1"


# important environment variables
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
google_api_keys_base = os.environ["GOOGLE_API_KEY"]
google_api_keys = [item.strip() for item in google_api_keys_base.split(';')]
os.environ["GOOGLE_API_KEY"] = random.choice(google_api_keys)



llm = init_chat_model("google_genai:gemini-2.5-flash")


# Modeling utils
# ==========================================================
# Trait-Based Functions
# ==========================================================

def logan_fecundity(T, a=0.378, b=0.173, Tmax=40.0, c=2.97):
    F_T = a * np.exp(b * T) - np.exp(b * Tmax - ((Tmax - T) / c))
    F_T = np.where((T > Tmax) | (F_T < 0), 0, F_T)
    return F_T

def dev_rate_egg(T):   return max(0, -0.0009*T**2 + 0.048*T - 0.345)
def dev_rate_larva(T): return max(0, -0.0007*T**2 + 0.039*T - 0.32)
def dev_rate_pupa(T):  return max(0, -0.0005*T**2 + 0.026*T - 0.2)

def surv_rate_egg(T):   return 1.01 * np.exp(-0.5 * ((T - 24.5) / 4.8)**2)
def surv_rate_larva(T): return 0.95 * np.exp(-0.5 * ((T - 27) / 3.5)**2)
def surv_rate_pupa(T):  return 0.93 * np.exp(-0.5 * ((T - 26.8) / 3.8)**2)

def mort_rate_adult(T):
    base = 0.0207
    a = 0.000603
    return base + a * (T - 26)**2

def compute_suitability(T):
    if np.isnan(T) or np.isinf(T) or (T < 0) or (T > 50):
        return 0.0
    
    time_step = 6
    F  = logan_fecundity(T)
    dE = dev_rate_egg(T)
    dL = dev_rate_larva(T)
    dP = dev_rate_pupa(T)
    sE = surv_rate_egg(T)
    sL = surv_rate_larva(T)
    sP = surv_rate_pupa(T)
    muA = mort_rate_adult(T)
    
    M = np.array([
        [-dE,      0,       0,        F],
        [sE*dE,  -dL,      0,        0],
        [0,      sL*dL,   -dP,       0],
        [0,       0,      sP*dP,   -muA]
    ]) * time_step
    
    eigs = np.linalg.eigvals(M)
    lambda_max = np.max(np.real(eigs))
    return max(0, lambda_max)


compute_suitability_vec = np.vectorize(compute_suitability)


def epi_prob(x):
    return np.where(x <= 1, 0, 1 - 1/x)


def fast_extract_pixel_values(path, band_number, coordinates):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        if band_number < 1 or band_number > src.count:
            raise ValueError(f"Invalid band number {band_number}, raster has {src.count} bands.")

        band = src.read(band_number)  # 2D array
        nodata = src.nodata
        height, width = band.shape
        
        # safe per-band scale / offset handling
        scales = getattr(src, "scales", None)
        if scales and 1 <= band_number <= len(scales):
            scale = scales[band_number - 1] if scales[band_number - 1] is not None else 1.0
        else:
            scale = 1.0

        offsets = getattr(src, "offsets", None)
        if offsets and 1 <= band_number <= len(offsets):
            offset = offsets[band_number - 1] if offsets[band_number - 1] is not None else 0.0
        else:
            offset = 0.0

        # Convert (lon, lat) to (row, col) using list comprehension
        rows_cols = [src.index(lon, lat) for lon, lat in coordinates]
        rows = np.array([r for r, _ in rows_cols])
        cols = np.array([c for _, c in rows_cols])

        # Initialize with NaNs
        values = np.full(len(rows), np.nan)

        # Bounds check
        in_bounds = (
            (rows >= 0) & (rows < height) &
            (cols >= 0) & (cols < width)
        )

        valid_rows = rows[in_bounds]
        valid_cols = cols[in_bounds]

        # Extract values using NumPy advanced indexing
        # extracted = band[valid_rows, valid_cols]
        extracted = band[valid_rows.astype(int), valid_cols.astype(int)]

        if nodata is not None:
            extracted = np.where(extracted == nodata, np.nan, extracted)

        # Assign back to full array
        values[in_bounds] = extracted
        values_list = values.tolist()
        values_list = [(v * scale + offset) if v is not None else None for v in values_list] # apply scale and offset

    return values_list



# clipping rasters tu study area
def clip_rasters_to_study_area(
    input_folder: str = "data/environmental",
    output_folder: str = "data/clipped",
    mask_path: str = "data/study_area.geojson"
    ):

    KNOWN_MISSING_RASTER_CRS = None
    shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Load mask once and dissolve to a single (multi)polygon for speed/stability
    mask_gdf = gpd.read_file(mask_path)
    mask_dissolved = mask_gdf.dissolve()    # single row
    mask_geom = mask_dissolved.geometry.values[0]       # (multi)polygon
    mask_geom_mapping = [mapping(mask_geom)]            # GeoJSON-like

    input_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    output_files = sorted(glob.glob(os.path.join(output_folder, "*.tif")))

    for input_file in input_files:
        file_name = os.path.basename(input_file)
        out_path = os.path.join(output_folder, file_name)
        if out_path in sorted(output_files):
            print(f"Skipping already processed file: {file_name}")
            continue
    
        # Lazy read (helps with large rasters)
        r = rxr.open_rasterio(input_file, masked=True)
    
        # — Preference 1: Vector → Raster CRS when raster has CRS —
        if r.rio.crs is not None: # pyright: ignore[reportAttributeAccessIssue]
            # Reproject vector to raster CRS if needed
            if mask_gdf.crs != r.rio.crs: # pyright: ignore[reportAttributeAccessIssue]
                mask_for_this = mask_dissolved.to_crs(r.rio.crs) # pyright: ignore[reportAttributeAccessIssue]
                mask_mapping = [mapping(mask_for_this.geometry.values[0])]
            else:
                mask_mapping = mask_geom_mapping
    
            # Clip and write
            clipped = r.rio.clip(mask_mapping, r.rio.crs) # pyright: ignore[reportAttributeAccessIssue]
        
        else:
            # — Preference 2: Raster → Vector CRS when raster is missing a CRS —
            # Decide what CRS to stamp on the raster so we can operate safely
            crs_to_write = KNOWN_MISSING_RASTER_CRS or mask_gdf.crs
            if crs_to_write is None:
                raise ValueError(
                    f"No CRS on raster and vector has no CRS either for {file_name}. "
                    "Please specify KNOWN_MISSING_RASTER_CRS."
                )
    
            # Stamp CRS (this does NOT reproject; it declares what the raster already is)
            r = r.rio.write_crs(crs_to_write) # pyright: ignore[reportAttributeAccessIssue]
    
            # Make sure mask is in the same CRS (usually it already is)
            if mask_gdf.crs != r.rio.crs:
                mask_for_this = mask_dissolved.to_crs(r.rio.crs)
                mask_mapping = [mapping(mask_for_this.geometry.values[0])]
            else:
                mask_mapping = mask_geom_mapping
    
            try:
                clipped = r.rio.clip(mask_mapping, r.rio.crs)
            except Exception as e:
                print(f"Skip (no overlap or clip error): {file_name} -> {e}")
                continue
    
    
        # Decide dtype & predictor based on the actual array dtype
        is_float = np.issubdtype(clipped.dtype, np.floating)
    
        if is_float:
            # ensure a float nodata and float32 dtype (so PREDICTOR=3 is allowed)
            if clipped.rio.nodata is None or not np.isnan(clipped.rio.nodata):
                clipped = clipped.rio.write_nodata(np.nan)
            if clipped.dtype != np.float32:
                clipped = clipped.astype("float32")
            pred = 3
            out_dtype = "float32"
        else:
            # integer path: ensure integer nodata and PREDICTOR=2
            pred = 2
            nd = clipped.rio.nodata
            # choose a safe integer nodata compatible with the dtype range
            if nd is None or isinstance(nd, float):
                if clipped.dtype == np.int16:
                    nd_value = -32768
                elif clipped.dtype == np.int32:
                    nd_value = -2147483648
                elif clipped.dtype == np.uint16:
                    nd_value = 0
                elif clipped.dtype == np.uint8:
                    nd_value = 0
                else:
                    # generic fallback
                    nd_value = -9999
                clipped = clipped.rio.write_nodata(nd_value)
            out_dtype = str(clipped.dtype)
    
        # Write with explicit dtype & matching predictor
        clipped.rio.to_raster(
            out_path,
            dtype=out_dtype,          # <- force GDAL to use the dtype we intend
            compress="deflate",
            predictor=pred,           # 3 only with float32/64; 2 for integers
            zlevel=6,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
    
    
        print(f"Processed: {file_name}")
        output_files.append(out_path)  # avoid re-processing if rerun
        
        # close any open raster handle, if present
        try:
            r.close() # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            pass
    
        # delete names without risking NameError
        try:
            del clipped, r
        except NameError:
            pass



# GBIF Utils

def get_taxon_key(species_name: str) -> int:
    """Get GBIF taxonKey for a species name"""
    url = f"{GBIF_API}/species/match"
    r = requests.get(url, params={"name": species_name})
    r.raise_for_status()
    data = r.json()
    if "usageKey" not in data:
        raise ValueError(f"Species {species_name} not found in GBIF")
    return data["usageKey"]


def define_study_area(countries: List[str], output_path:str = "./data/study_area.geojson"):
    world_shp = gpd.read_file('world-administrative-boundaries.geojson')
    study_area_shp = world_shp[world_shp['name'].isin(countries)]
    # print(study_area_shp)
    # print(study_area_shp.is_empty.sum(), "empty geometries")
    if study_area_shp.crs is None:
        study_area_shp = study_area_shp.set_crs("EPSG:4326")
    study_area_shp = study_area_shp[~study_area_shp.geometry.is_empty]
    # Save to GeoJSON
    study_area_shp.to_file(output_path, driver="GeoJSON")
    # study_area_shp.plot()
    return output_path


# Define tool: download WorldClim temperature rasters
@tool
def download_worldclim_temp(period: str, resolution: str = "10m", out_dir: str = "./data") -> List[str]:
    """
    Download WorldClim temperature rasters for a given period.
    
    Args:
        period (str): "monthly", "annual", "bioclim", or "elevation".
        resolution (str): Spatial resolution ("10m", "5m", "2.5m", "30s").
        out_dir (str): Directory to save the rasters.
    
    Returns:
        List[str]: Paths to the downloaded raster files.
    """
    base_url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
    
    if period == "monthly":
        file_name = f"wc2.1_{resolution}_tavg.zip"
    elif period == "annual":
        file_name = f"wc2.1_{resolution}_tavg_ann.zip"
    elif period == "bioclim":
        file_name = f"wc2.1_{resolution}_bio.zip"
    elif period == "elevation":
        file_name = f"wc2.1_{resolution}_elev.zip"
    else:
        raise ValueError("Invalid period. Choose from 'monthly', 'annual', 'bioclim', 'elevation'.")

    url = f"{base_url}/{file_name}"
    os.makedirs(out_dir, exist_ok=True)
    local_zip = os.path.join(out_dir, file_name)
    
    # Download
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    # Collect GeoTIFFs
    tifs = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".tif")]
    
    return tifs



# gbif data download tool
@tool
def download_gbif_occurrences(
        species: str,
        country: Optional[str] = None,
        limit: int = 500,
        n_absences: Optional[int] = None,
        output_dir: str = "./data/gbif_data"
    ) -> Dict[str, Any]:
    """
    Download species occurrence records from GBIF, add presence & pseudo-absence points,
    and save to CSV.

    Args:
        species: Scientific name of the species (e.g., "Papio anubis").
        country: Optional 2-letter ISO country code .
        limit: Number of presence records to fetch (max 300,000 via paging).
        n_absences: Number of pseudo-absence/background points to generate. 
                    If None, defaults to same as presence count.
        output_dir: Directory where the CSV will be stored.

    Returns:
        A dictionary with the path to the saved CSV file and the number of records.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Fetch PRESENCE data from GBIF
    # -----------------------------
    records = []
    offset = 0
    batch_size = 300  # GBIF max per request

    while offset < limit:
        params = {
            "scientificName": species,
            "limit": min(batch_size, limit - offset),
            "offset": offset,
            "hasGeospatialIssue": False,
            "hasCoordinate": True
        }
        if country:
            params["country"] = country.upper()

        url = "https://api.gbif.org/v1/occurrence/search"
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"GBIF request failed: {response.status_code}")

        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        for r in results:
            if r.get("decimalLatitude") and r.get("decimalLongitude"):
                records.append({
                    "species": r.get("species"),
                    "lat": r.get("decimalLatitude"),
                    "lon": r.get("decimalLongitude"),
                    "country": r.get("country"),
                    "eventDate": r.get("eventDate"),
                    "basisOfRecord": r.get("basisOfRecord"),
                    "presence": 1
                })

        offset += batch_size

    if not records:
        raise ValueError(f"No records found for {species} (country={country})")

    df = pd.DataFrame(records)

    # -----------------------------
    # Generate ABSENCE (background) data
    # -----------------------------
    if n_absences is None:
        n_absences = len(df)


    if country:
        NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(NE_URL)
        geom = world[world["ISO_A2"] == country.upper()].geometry.values[0]
    else:
        # Global bounding box
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        geom = world.unary_union  # all countries merged

    absences = []
    minx, miny, maxx, maxy = geom.bounds
    while len(absences) < n_absences:
        rand_x = np.random.uniform(minx, maxx)
        rand_y = np.random.uniform(miny, maxy)
        pt = Point(rand_x, rand_y)
        if geom.contains(pt):
            absences.append({"species": species, "lat": rand_y, "lon": rand_x,
                             "country": country or "GLOBAL",
                             "eventDate": None,
                             "basisOfRecord": "pseudoAbsence",
                             "presence": 0})

    df_abs = pd.DataFrame(absences)

    # -----------------------------
    # Combine presence + absence
    # -----------------------------
    df_all = pd.concat([df, df_abs], ignore_index=True)

    # Save to CSV
    out_csv = os.path.join(output_dir, f"{species.replace(' ', '_')}_{country or 'WORLD'}_with_absences.csv")
    df_all.to_csv(out_csv, index=False)

    return {"csv_path": out_csv, "records": len(df_all), "presence": len(df), "absences": len(df_abs)}



# download shapefiles
@tool
def download_shapefile(
        iso3: str,
        level: int = 0,
        output_dir: str = "./data/shapefiles"
    ):
    """
    Download and save GADM boundaries as GeoJSON.
    
    Args:
        iso3: ISO3 country code (e.g., "KEN").
        level: Administrative level (0 = country, 1 = provinces, 2 = districts, etc.)
        output_dir: Directory where the GeoJSON will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)

    url = GADM_JSON_URL.format(iso3=iso3.upper(), level=level)
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download GADM JSON for {iso3} at level {level}")

    out_path = os.path.join(output_dir, f"gadm41_{iso3.upper()}_{level}.json")
    with open(out_path, "wb") as f:
        f.write(response.content)

    # Validate with GeoPandas
    try:
        gdf = gpd.read_file(out_path)
    except Exception as e:
        raise ValueError(f"Downloaded file is not a valid GeoJSON: {e}")

    return {"shapefile": out_path, "features": len(gdf)}



# suitability model tool
@tool
def run_suitability_model(temp_rast_path: str, output_path: str) -> str:
    """Compute suitability from a temperature raster using the Logan model
    and epidemic probability normalization. Saves and displays the result."""
    # Load raster
    with rasterio.open(temp_rast_path) as src:
        temp_rast = src.read(1)
        profile = src.profile

    # Replace NaN with 0
    temp_rast = np.where(np.isnan(temp_rast), 0, temp_rast)

    # Compute suitability
    lambda_raster = compute_suitability_vec(temp_rast)
    lambda_norm_raster = epi_prob(lambda_raster)

    # Save output raster
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(lambda_norm_raster.astype(np.float32), 1)

    height, width = lambda_norm_raster.shape
    aspect = width / height
    fig_height = 8  # base size
    fig_width = fig_height * aspect

    # Plot
    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(lambda_norm_raster, cmap="viridis", aspect="auto")
    plt.title("Normalized Epidemic Probability (λmax)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()
    unique_name = f"{uuid.uuid4().hex}.png"
    plt.savefig(os.path.join("uploads", unique_name), dpi=300)
    shutil.copy(output_path, f"uploads/{unique_name}.tif")

    return {"path": output_path, "file_name": unique_name}



# niche modeling tool

@tool
def run_ecological_niche_model(
        species_name: str,
        occurrence_data_path: str,
        environmental_data_path: str,
        output_raster_path: str,
        bio_predictors: List[int] = [1,2,3],
        elevation: bool = True,
        resolution: str = "10m",
        study_area: List[str] = ['Kenya']
    ) -> str:
    """
    Trains a Maxent ecological niche model and applies it to environmental rasters
    for a specified species.

    Args:
        species_name: The name of the species to model (e.g., 'Busseola_fusca').
                      This is used to identify the correct column in the occurrence data.
        occurrence_data_path: Path to the CSV file with species occurrence and background data.
        environmental_data_path: Path to the folder containing environmental raster files.
        output_raster_path: Path to save the output GeoTIFF raster.
        bio_predictors: List of Bioclim variable id to include, a number between 1 and 19.
        elevation: boolean to specify if we should include or not elevation in the modeling.
        resolution (str): Spatial resolution ("10m", "5m", "2.5m", "30s").
        study_area: List of countries names ['Cameroon', 'Kenya'] for instance
    Returns:
        Path to the output
    """
    # -----------------------------
    # Load occurrence/background data
    # -----------------------------
    df_occ = pd.read_csv(occurrence_data_path)
    df_occ = df_occ.dropna(subset=["lon", "lat"])  # must have coords
    
    species_presence_column = "presence"
    occurrence_data = df_occ[df_occ[species_presence_column] == 1].copy()
    background_data = df_occ[df_occ[species_presence_column] == 0].copy()

    # Coordinates for raster extraction
    coordinates = list(zip(occurrence_data["lon"], occurrence_data["lat"]))
    band_number = 1

    # -----------------------------
    # Define predictors
    # -----------------------------
    predictors = [f"{resolution}_bio_{i}" for i in bio_predictors]
    if elevation:
        predictors.append(f"{resolution}_elev")
        
    # -----------------------------
    # Clip rasters to study area
    # -----------------------------
    define_study_area(countries=study_area)
    clipped_data_folder = "./data/clipped"
    clip_rasters_to_study_area(
        input_folder=environmental_data_path,
        output_folder=clipped_data_folder,
        mask_path="./data/study_area.geojson"
    )

    # -----------------------------
    # Extract raster values
    # -----------------------------
    for file in sorted(os.listdir(clipped_data_folder)):
        if not file.endswith(".tif"):
            continue
        file_path = os.path.join(clipped_data_folder, file)
        filename, _ = os.path.splitext(file)
        for predictor in predictors:
            if predictor in filename:
                # safe assignment
                occurrence_data.loc[:, predictor] = fast_extract_pixel_values(file_path, band_number, coordinates)
                background_data.loc[:, predictor] = fast_extract_pixel_values(file_path, band_number, coordinates)

    occurrence_points = occurrence_data[predictors]
    background_points = background_data[predictors]

    X = pd.concat([occurrence_points, background_points], ignore_index=True)
    y = pd.Series([1] * len(occurrence_points) + [0] * len(background_points))

    # -----------------------------
    # Handle NaNs in predictors
    # -----------------------------
    mask = ~X.isna().any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    if X_clean.empty:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = y
        if X_clean.empty:
            raise ValueError("No valid samples remain after NaN handling. "
                             "Check that your study area overlaps with raster coverage.")

    # -----------------------------
    # Train Maxent model
    # -----------------------------
    maxent_model = ela.MaxentModel()
    maxent_model.fit(X_clean, y_clean)

    # -----------------------------
    # Apply model to rasters
    # -----------------------------
    rasters = [os.path.join(clipped_data_folder, f"wc2.1_{p}.tif") for p in predictors]
    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    ela.apply_model_to_rasters(maxent_model, rasters, output_raster_path)

    # -----------------------------
    # Read raster data for plotting
    # -----------------------------
    with rasterio.open(output_raster_path) as src:
        raster_data = src.read(1).astype(float)
        raster_data[raster_data == src.nodata] = np.nan  # mask NoData

    # Plot result
    plt.figure(figsize=(8, 6))
    im = plt.imshow(raster_data, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Suitability")
    plt.title(f"Ecological Niche Model - {species_name}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    unique_name = f"{uuid.uuid4().hex}"
    plt.savefig(os.path.join("uploads", unique_name), dpi=300)
    shutil.copy(output_raster_path, f"uploads/{unique_name}.tif")

    return {"path": output_raster_path, "file_name": unique_name + ".png"}



@tool
def get_lulc_data(
    bbox: list[float], 
    year: int = 2023, 
    output_path: str = "lulc_data.tif"
) -> str:
    """
    Downloads Land Use Land Cover (LULC) data for a specific area and year 
    using the Esri 10m Annual LULC dataset via Microsoft Planetary Computer.
    
    Args:
        bbox: A list of 4 floats representing the bounding box [min_lon, min_lat, max_lon, max_lat].
        year: The year of the data (available 2017-2023).
        output_path: The filename to save the resulting GeoTIFF.
    """
    try:
        # Initialize the STAC API client
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            ignore_conformance=True,
        )

        # Search for the LULC collection
        search = catalog.search(
            collections=["io-lulc-annual-v02"],
            bbox=bbox,
            datetime=str(year),
        )

        items = list(search.get_items())
        if not items:
            return f"No LULC data found for the year {year} in this region."

        # Sign the asset to get a temporary access token
        selected_item = items[0]
        signed_item = planetary_computer.sign(selected_item)
        asset_url = signed_item.assets["data"].href

        # Load and clip the data to the user's specific bounding box
        data = rioxarray.open_rasterio(asset_url)
        clipped_data = data.rio.clip_box(*bbox)

        # Save to local file
        clipped_data.rio.to_raster(output_path)

        return f"Successfully downloaded LULC data for {year} to {output_path}."

    except Exception as e:
        return f"An error occurred: {str(e)}"


@tool
def get_bounding_box_for_country(
    country_name: str, 
) -> str:
    """
    Get bounding box for a given country, returns bbox
    """
    
    world = gpd.read_file('world-administrative-boundaries.geojson')
    country_gdf = world[world['name'] == country_name]
    if country_gdf.empty:
        return f"Country '{country_name}' not found."
    bbox = list(country_gdf.total_bounds) 
    return bbox


@tool
def extract_environmental_values(
    occurrence_csv_path: str,
    environmental_folder: str,
    study_area: List[str],
    predictors: List[str],
    resolution: str = "10m",
    output_csv_path: Optional[str] = None
) -> str:
    """
    Extract environmental values from raster files to occurrence points.
    This must be done BEFORE collinearity analysis and modeling.
    
    Args:
        occurrence_csv_path: Path to occurrence CSV file
        environmental_folder: Path to environmental raster folder
        study_area: List of country names for clipping
        predictors: List of predictor names (e.g., ['bio_1', 'bio_2', 'elev'])
        resolution: Resolution string (e.g., '10m')
        output_csv_path: Optional output path (default adds '_with_env' suffix)
    
    Returns:
        Path to CSV with environmental values extracted
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Load occurrence data
    df_occ = pd.read_csv(occurrence_csv_path)
    df_occ = df_occ.dropna(subset=["lon", "lat"])
    
    if "presence" not in df_occ.columns:
        return "Error: CSV must have 'presence' column (1 for presence, 0 for background)"
    
    # Define study area and clip rasters
    print(f"Defining study area for: {study_area}")
    define_study_area(countries=study_area, output_path="./data/study_area.geojson")
    
    clipped_folder = "./data/clipped"
    print(f"Clipping rasters to study area...")
    clip_rasters_to_study_area(
        input_folder=environmental_folder,
        output_folder=clipped_folder,
        mask_path="./data/study_area.geojson"
    )
    
    # Prepare coordinates for extraction
    coordinates = list(zip(df_occ["lon"], df_occ["lat"]))
    
    # Extract values for each predictor
    print(f"Extracting values for {len(predictors)} predictors...")
    
    for predictor in predictors:
        # Find the matching raster file
        raster_file = None
        for file in os.listdir(clipped_folder):
            if file.endswith(".tif"):
                # Handle different naming patterns
                if predictor in file or f"wc2.1_{resolution}_{predictor}" in file:
                    raster_file = os.path.join(clipped_folder, file)
                    break
        
        if not raster_file:
            print(f"Warning: Could not find raster for predictor: {predictor}")
            df_occ[predictor] = np.nan
            continue
        
        try:
            # Extract values
            values = fast_extract_pixel_values(raster_file, 1, coordinates)
            df_occ[predictor] = values
            print(f"  ✓ Extracted: {predictor}")
        except Exception as e:
            print(f"  ✗ Error extracting {predictor}: {str(e)}")
            df_occ[predictor] = np.nan
    
    # Save to CSV
    if output_csv_path is None:
        base, ext = os.path.splitext(occurrence_csv_path)
        output_csv_path = f"{base}_with_env{ext}"
    
    df_occ.to_csv(output_csv_path, index=False)
    print(f"\nSaved extracted data to: {output_csv_path}")
    print(f"Rows: {len(df_occ)}, Presence: {(df_occ['presence'] == 1).sum()}, Absence: {(df_occ['presence'] == 0).sum()}")
    
    return output_csv_path


@tool
def analyze_predictor_colinearity(
    occurrence_data_path: str, 
    predictors: List[str],
    threshold: float = 0.8,
    output_plot_path: Optional[str] = None
) -> str:
    """
    Analyzes and plots colinearity (Correlation Matrix) for the selected predictors.
    Run this AFTER extracting environmental values to occurrence points.
    
    Args:
        occurrence_data_path: Path to CSV with environmental values already extracted
        predictors: List of predictor names to analyze
        threshold: Correlation threshold above which variables are considered highly correlated
    
    Returns:
        Analysis results with recommendations
    """
    
    # Load data
    df = pd.read_csv(occurrence_data_path)
    
    # Check if predictors exist in the data
    available_predictors = [p for p in predictors if p in df.columns]
    missing = [p for p in predictors if p not in df.columns]
    
    if missing:
        return f"Error: These predictors are missing from the data: {missing}. Run extraction first."
    
    if len(available_predictors) < 2:
        return "Need at least 2 predictors for collinearity analysis."
    
    # Filter to only presence points for better analysis
    df_presence = df[df["presence"] == 1]
    if len(df_presence) < 10:
        print("Warning: Few presence points (<10), using all data including background.")
        df_analysis = df
    else:
        df_analysis = df_presence
    
    # Calculate correlation matrix
    corr_matrix = df_analysis[available_predictors].corr()
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    recommendations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                high_corr_pairs.append((var1, var2, corr_value))
                
                # Suggest which one to keep based on correlation with other variables
                # (simple heuristic: keep the one with lower average correlation with others)
                avg_corr_1 = abs(corr_matrix[var1]).mean()
                avg_corr_2 = abs(corr_matrix[var2]).mean()
                
                if avg_corr_1 <= avg_corr_2:
                    recommendations.append(f"Keep '{var1}' (avg corr: {avg_corr_1:.3f}), remove '{var2}' (r = {corr_value:.3f})")
                else:
                    recommendations.append(f"Keep '{var2}' (avg corr: {avg_corr_2:.3f}), remove '{var1}' (r = {corr_value:.3f})")
    
    # # Plot
    # plt.figure(figsize=(max(8, len(predictors)), max(6, len(predictors) * 0.8)))
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # sns.heatmap(
    #     corr_matrix, 
    #     annot=True, 
    #     cmap='coolwarm', 
    #     fmt=".2f",
    #     mask=mask,
    #     center=0,
    #     square=True,
    #     linewidths=0.5,
    #     cbar_kws={"shrink": 0.8}
    # )
    # plt.title(f"Predictor Correlation Matrix (Threshold: {threshold})")
    # plt.tight_layout()
    # plt.show()

    # Plot
    plt.figure(figsize=(max(8, len(predictors)), max(6, len(predictors) * 0.8)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f",
        mask=mask,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Predictor Correlation Matrix (Threshold: {threshold})")
    plt.tight_layout()

    # Save if output path provided
    print(f"#### OUTPUT PLOT PATH: {output_plot_path}")
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved collinearity heatmap to {output_plot_path}")
    else:
        unique_name = f"{uuid.uuid4().hex}.png"
        plt.savefig(f"./data/outputs/colinearity_analysis_{unique_name}", dpi=300, bbox_inches='tight')
        import shutil
        shutil.copy2(f"./data/outputs/colinearity_analysis_{unique_name}", "./uploads")
        print(f"Saved collinearity heatmap to {output_plot_path}")
        

    # plt.show()
    # plt.close()
    
    # Prepare summary message
    summary = []
    summary.append("=" * 60)
    summary.append("COLLINEARITY ANALYSIS RESULTS")
    summary.append("=" * 60)
    summary.append(f"Data points analyzed: {len(df_analysis)} ({len(df_presence)} presence)")
    summary.append(f"Predictors analyzed: {len(available_predictors)}")
    summary.append("")
    
    if high_corr_pairs:
        summary.append(f"FOUND {len(high_corr_pairs)} HIGHLY CORRELATED PAIRS (r > {threshold}):")
        summary.append("-" * 40)
        for var1, var2, corr in high_corr_pairs:
            summary.append(f"  {var1} ↔ {var2}: r = {corr:.3f}")
        
        summary.append("")
        summary.append("RECOMMENDATIONS:")
        summary.append("-" * 40)
        for rec in recommendations[:5]:  # Show top 5 recommendations
            summary.append(f"  • {rec}")
        
        if len(recommendations) > 5:
            summary.append(f"  ... and {len(recommendations) - 5} more")
        
        summary.append("")
        summary.append("SUGGESTED WORKFLOW:")
        summary.append("1. Remove one variable from each highly correlated pair")
        summary.append("2. Re-run collinearity analysis with remaining variables")
        summary.append("3. Proceed to modeling with non-correlated predictors")
    else:
        summary.append(f"✓ No highly correlated pairs found (all r ≤ {threshold})")
        summary.append("✓ You can proceed with all predictors in your model")
    
    summary.append("=" * 60)
    
    return "\n".join(summary)



@tool
def download_lulc_stable(query: str, year: int = 2021, scale: int = 10, output_path: str = "lulc.tif") -> str:
    """
    Downloads 10m LULC data for 2021. 
    Handles local downloads and switches to Drive for large areas automatically.
    """
    try:
        # 1. Initialize
        ee.Initialize(project="gen-lang-client-0359288668")
        
        # 2. Robust Region Search
        # Uses stringContains for flexibility (Addis Ababa vs Addis Ababa City)
        roi = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
            ee.Filter.stringContains('ADM1_NAME', query)
        )
        if roi.size().getInfo() == 0:
            roi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
                ee.Filter.stringContains('ADM0_NAME', query)
            )
        
        if roi.size().getInfo() == 0:
            return f"Error: Could not find boundary for '{query}'"

        # 3. Load Data (ESA WorldCover 10m v200 for 2021)
        img = ee.Image("ESA/WorldCover/v200/2021").clip(roi)
        region = roi.geometry().bounds()
        region_coords = region.getInfo()['coordinates']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 4. Hybrid Export Logic
        try:
            print(f"Attempting direct download for {query} at {scale}m...")
            # Using download_ee_image which is more stable for local GeoTIFFs
            geemap.download_ee_image(
                image=img,
                filename=output_path,
                region=region_coords,
                scale=scale,
                crs="EPSG:4326"
            )
            return f"Successfully saved locally: {output_path}"
        
        except Exception as e:
            # If local download fails (usually due to 10m res over a city being too big)
            print("Local download limit hit. Switching to Google Drive Batch Export...")
            
            task_name = f"LULC_{query.replace(' ', '_')}_{year}"
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=task_name,
                folder='GEE_LULC_Data',
                fileNamePrefix=os.path.basename(output_path).replace('.tif', ''),
                scale=scale,
                region=region,
                maxPixels=1e13 # Support for up to 10 trillion pixels
            )
            task.start()
            
            # Monitoring loop
            while task.active():
                print(f"Task status: {task.status()['state']}... (waiting 30s)")
                time.sleep(30)
            
            if task.status()['state'] == 'COMPLETED':
                return f"Region too large for direct download. File exported to Google Drive folder 'GEE_LULC_Data' as {task_name}.tif"
            else:
                return f"Drive Export failed: {task.status()['error_message']}"

    except Exception as e:
        return f"Tool Error: {str(e)}"
    


@tool
def download_lulc_unified(
    country_name: str,
    year: int = 2021,
    output_dir: str = "./data/environmental",
    method: str = "auto"
) -> str:
    """
    Download Land Use Land Cover data using multiple possible sources.
    
    Args:
        country_name: Name of the country (e.g., 'Kenya')
        year: Year of LULC data (2017-2023 for Planetary Computer)
        output_dir: Directory to save the LULC raster
        method: 'planetary' (Microsoft Planetary Computer), 'gee' (Google Earth Engine), or 'auto'
    
    Returns:
        Path to downloaded LULC raster or error message
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get bounding box
    try:
        bbox = get_bounding_box_for_country(country_name)
        if isinstance(bbox, str) and "not found" in bbox:
            return f"Error: {bbox}"
    except Exception as e:
        return f"Error getting bounding box: {str(e)}"
    
    # Prepare output path
    output_path = os.path.join(output_dir, f"lulc_data.tif")
    
    # Method 1: Try Planetary Computer first (most reliable)
    if method in ["auto", "planetary"]:
        try:
            print("Attempting Planetary Computer download...")
            result = get_lulc_data(bbox, year, output_path)
            
            if "Successfully" in result or "tif" in result:
                print("✓ Planetary Computer download successful")
                return f"LULC downloaded from Planetary Computer: {output_path}"
            else:
                print(f"Planetary Computer returned: {result}")
        except Exception as e:
            print(f"Planetary Computer error: {str(e)}")
    
    # Method 2: Try Google Earth Engine (if configured)
    if method in ["auto", "gee"]:
        try:
            print("Attempting Google Earth Engine download...")
            # Note: This requires proper GEE setup
            result = download_lulc_stable(
                query=country_name,
                year=year,
                output_path=output_path
            )
            
            if "Successfully" in result or "saved" in result or "Drive" in result:
                print("✓ Google Earth Engine download initiated")
                return result
            else:
                print(f"Google Earth Engine returned: {result}")
        except Exception as e:
            print(f"Google Earth Engine error: {str(e)}")
    
    # Method 3: Falong tongue pmvllback - use existing LULC data if available
    potential_files = [
        os.path.join(output_dir, "lulc.tif"),
        os.path.join(output_dir, "lulc_data.tif"),
        os.path.join("./data", "lulc.tif")
    ]
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            print(f"✓ Using existing LULC file: {file_path}")
            # Copy to expected location
            import shutil
            shutil.copy2(file_path, output_path)
            return f"Using existing LULC file: {output_path}"
    
    return "Error: Could not download LULC data. Please check:\n" \
           "1. Internet connection\n" \
           "2. API keys for Planetary Computer or Google Earth Engine\n" \
           "3. Or provide your own LULC data in ./data/environmental/lulc_data.tif"



@tool
def complete_modeling_workflow_with_lulc(
    species: str = "Apis mellifera",
    country: str = "Kenya",
    include_lulc: bool = True,
    lulc_year: int = 2021,
    n_occurrences: int = 300,
    output_dir: str = "./data/outputs"
) -> Dict[str, Any]:
    """
    Complete workflow from data download to modeling with proper LULC integration.
    """
    import time
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    steps_log = []
    results = {}
    
    try:
        # Step 1: Download environmental data
        steps_log.append(f"{timestamp} - Step 1: Downloading environmental data")
        
        # Bioclimatic data
        bioclim_result = download_worldclim_temp(
            period="bioclim",
            resolution="10m",
            out_dir="./data/environmental"
        )
        steps_log.append(f"  Downloaded {len(bioclim_result)} bioclim files")
        
        # Elevation data
        elev_result = download_worldclim_temp(
            period="elevation",
            resolution="10m",
            out_dir="./data/environmental"
        )
        steps_log.append(f"  Downloaded elevation data")
        
        # LULC data (if requested)
        if include_lulc:
            lulc_result = download_lulc_for_modeling(
                country_name=country,
                resolution="10m",
                year=lulc_year,
                output_dir="./data/environmental",
                method="planetary"
            )
            steps_log.append(f"  Downloaded LULC data: {lulc_result}")
        else:
            lulc_result = None
            steps_log.append("  Skipping LULC download")
        
        # Step 2: Download occurrence data
        steps_log.append(f"\n{timestamp} - Step 2: Downloading occurrence data")
        gbif_result = download_gbif_occurrences(
            species=species,
            country=country[:2].upper() if len(country) > 2 else country,
            limit=n_occurrences,
            output_dir="./data/gbif_data"
        )
        steps_log.append(f"  Downloaded {gbif_result['records']} records")
        results["gbif_data"] = gbif_result["csv_path"]
        
        # Step 3: Extract environmental values
        steps_log.append(f"\n{timestamp} - Step 3: Extracting environmental values")
        
        # Define predictors based on what was downloaded
        bio_predictors = list(range(1, 20))  # All bioclim variables
        
        extracted_csv = extract_environmental_values(
            occurrence_csv_path=gbif_result["csv_path"],
            environmental_folder="./data/environmental",
            study_area=[country],
            predictors=[f"10m_bio_{i}" for i in bio_predictors] + 
                      ["10m_elev"] + 
                      (["10m_lulc"] if include_lulc else []),
            resolution="10m",
            output_csv_path=os.path.join(output_dir, f"{species.replace(' ', '_')}_with_env.csv")
        )
        steps_log.append(f"  Extracted environmental values to: {extracted_csv}")
        results["extracted_data"] = extracted_csv
        
        # Step 4: Analyze collinearity
        steps_log.append(f"\n{timestamp} - Step 4: Analyzing predictor collinearity")

        collinearity_plot_path = extracted_csv.replace('.csv', '_collinearity.png')

        predictors_for_collinearity = [f"10m_bio_{i}" for i in bio_predictors] + ["10m_elev"]
        if include_lulc:
            predictors_for_collinearity.append("10m_lulc")
    
        collinearity_result = analyze_predictor_colinearity.invoke({
            "occurrence_data_path": extracted_csv,
            "predictors": predictors,
            "threshold": 0.8,
            "output_plot_path": collinearity_plot_path   # new argument
        })
        steps_log.append("  Collinearity analysis complete")
        results["collinearity_plot"] = collinearity_plot_path
        
        # Step 5: Run ecological niche model with LULC
        steps_log.append(f"\n{timestamp} - Step 5: Running ecological niche model")
        
        model_output = os.path.join(output_dir, f"{species.replace(' ', '_')}_suitability.tif")
        
        model_results = run_ecological_niche_model_with_lulc(
            species_name=species,
            occurrence_data_path=extracted_csv,
            environmental_data_path="./data/environmental",
            output_raster_path=model_output,
            bio_predictors=bio_predictors,
            include_elevation=True,
            include_lulc=include_lulc,
            resolution="10m",
            study_area=[country]
        )

        if isinstance(model_results, str) and os.path.exists(model_results):
            steps_log.append(f"  Model successful: {model_output}")
            results["model_output"] = model_output
            # The PNG was already saved inside run_ecological_niche_model
            model_png = model_output.replace('.tif', '.png')
            results["model_plot"] = model_png if os.path.exists(model_png) else None
        else:
            steps_log.append(f"  Model returned: {model_results}")
            results["model_result"] = model_results
        
        if "error" in model_results:
            steps_log.append(f"  Model failed: {model_results['error']}")
            results["error"] = model_results["error"]
        else:
            steps_log.append(f"  Model successful: {model_output}")
            results.update(model_results)
        
        # Final summary
        steps_log.append(f"\n{'='*60}")
        steps_log.append("WORKFLOW COMPLETE")
        steps_log.append(f"{'='*60}")
        
        results["success"] = "error" not in results
        results["log"] = steps_log
        results["timestamp"] = timestamp
        
        # Save workflow log
        log_file = os.path.join(output_dir, f"workflow_log_{timestamp}.txt")
        with open(log_file, 'w') as f:
            f.write("\n".join(steps_log))
        
        results["log_file"] = log_file
        
    except Exception as e:
        steps_log.append(f"\nERROR: {str(e)}")
        import traceback
        steps_log.append(f"Traceback: {traceback.format_exc()}")
        
        results = {
            "success": False,
            "error": str(e),
            "log": steps_log,
            "timestamp": timestamp
        }
    
    return results



@tool
def execute_modeling_workflow(
    species: str = "Apis mellifera",
    country: str = "Kenya",
    predictors: List[str] = None,
    n_occurrences: int = 300
) -> str:
    """
    Execute the complete ecological niche modeling workflow.
    
    Args:
        species: Species scientific name
        country: Country name
        predictors: List of predictors to use (if None, uses defaults)
        n_occurrences: Number of occurrence records to download
    
    Returns:
        Summary of workflow execution
    """
    import time
    
    if predictors is None:
        predictors = ["bio_1", "bio_2", "bio_3", "bio_4", "bio_5", 
                     "bio_6", "bio_7", "bio_8", "bio_9", "bio_10",
                     "bio_11", "bio_12", "bio_13", "bio_14", "bio_15",
                     "bio_16", "bio_17", "bio_18", "bio_19", "elev"]
    
    steps = []
    
    try:
        # Step 1: Download environmental data - FIXED
        steps.append("Step 1: Downloading WorldClim bioclimatic data...")
        bioclim_files = download_worldclim_temp.invoke({
            "period": "bioclim", 
            "resolution": "10m", 
            "out_dir": "./data/environmental"
        })
        steps.append(f"  ✓ Downloaded bioclim files")
        
        time.sleep(1)  # Brief pause
        
        # Step 2: Download elevation - FIXED
        steps.append("Step 2: Downloading elevation data...")
        elev_files = download_worldclim_temp.invoke({
            "period": "elevation",
            "resolution": "10m",
            "out_dir": "./data/environmental"
        })
        steps.append(f"  ✓ Downloaded elevation data")
        
        # Step 3: Download LULC - FIXED
        steps.append("Step 3: Downloading Land Use Land Cover data...")
        lulc_result = download_lulc_unified.invoke({
            "country_name": country,
            "year": 2021,
            "output_dir": "./data/environmental",
            "method": "planetary"
        })
        steps.append(f"  ✓ {lulc_result[:100]}...")
        
        # Step 4: Download shapefile - FIXED
        steps.append("Step 4: Downloading shapefiles...")
        iso3 = "KEN"  # You might want to add ISO3 lookup
        shape_result = download_shapefile.invoke({
            "iso3": iso3,
            "level": 0,
            "output_dir": "./data/shapefiles"
        })
        steps.append(f"  ✓ Downloaded shapefile with {shape_result['features']} features")
        
        # Step 5: Download GBIF data - FIXED
        steps.append(f"Step 5: Downloading {n_occurrences} {species} occurrences...")
        gbif_result = download_gbif_occurrences.invoke({
            "species": species,
            "country": "KE",  # ISO2 code
            "limit": n_occurrences,
            "output_dir": "./data/gbif_data"
        })
        steps.append(f"  ✓ Downloaded {gbif_result['records']} records ({gbif_result['presence']} presence, {gbif_result['absences']} absence)")
        
        # Step 6: Extract environmental values - FIXED
        steps.append("Step 6: Extracting environmental values to occurrence points...")
        extracted_csv = extract_environmental_values.invoke({
            "occurrence_csv_path": gbif_result["csv_path"],
            "environmental_folder": "./data/environmental",
            "study_area": [country],
            "predictors": predictors,
            "resolution": "10m"
        })
        steps.append(f"  ✓ Extracted to: {extracted_csv}")
        
        # Step 7: Analyze collinearity - FIXED
        steps.append("Step 7: Analyzing predictor collinearity...")
        collinearity_result = analyze_predictor_colinearity.invoke({
            "occurrence_data_path": extracted_csv,
            "predictors": predictors,
            "threshold": 0.8
        })
        steps.append("  ✓ Collinearity analysis complete")
        
        return "\n".join(steps) + "\n\n" + collinearity_result + \
               "\n\nPlease review the collinearity results and specify which predictors to use for modeling."
        
    except Exception as e:
        steps.append(f"❌ Error in workflow: {str(e)}")
        return "\n".join(steps)
    


@tool
def download_lulc_for_modeling(
    country_name: str,
    resolution: str = "10m",
    year: int = 2021,
    output_dir: str = "./data/environmental",
    method: str = "planetary"
) -> str:
    """
    Download LULC data and rename it to match WorldClim naming convention.
    This ensures proper integration with ecological niche models.
    
    Args:
        country_name: Country name (e.g., 'Kenya')
        resolution: Resolution string matching WorldClim (e.g., '10m')
        year: Year of LULC data
        output_dir: Directory to save the LULC raster
        method: 'planetary', 'gee', or 'auto' (default 'planetary')
    
    Returns:
        Path to the LULC raster in WorldClim format
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get bounding box
    bbox = get_bounding_box_for_country.invoke({
        "country_name": country_name
    })
    if isinstance(bbox, str):
        return f"Error: {bbox}"
    
    # Determine final output path
    lulc_worldclim_name = f"wc2.1_{resolution}_lulc.tif"
    final_output = os.path.join(output_dir, lulc_worldclim_name)
    
    if method == "planetary":
        # Try Planetary Computer directly
        result = get_lulc_data.invoke({
            "bbox": bbox,
            "year": year,
            "output_path": final_output
        })
        if "Error" not in result and "failed" not in result.lower():
            # Already saved with the correct name, just verify
            pass
        else:
            return f"LULC download failed: {result}"
    elif method == "gee":
        # Try Google Earth Engine directly
        result = download_lulc_stable.invoke({
            "query": country_name,
            "year": year,
            "scale": 10,
            "output_path": final_output
        })
        if "Error" not in result and "failed" not in result.lower():
            pass
        else:
            return f"LULC download failed: {result}"
    elif method == "auto":
        # Use unified downloader which has fallback logic
        result = download_lulc_unified.invoke({
            "country_name": country_name,
            "year": year,
            "output_dir": output_dir,
            "method": "auto"
        })
        # The unified downloader returns a message that may contain the final path.
        # We need to extract the actual file path.
        if "Error" in result or "failed" in result.lower():
            return f"LULC download failed: {result}"
        # Try to find the path in the message (e.g., "LULC downloaded from Planetary Computer: ./data/environmental/lulc_data.tif")
        import re
        match = re.search(r'(\./[^\s]+\.tif)', result)
        if match:
            temp_path = match.group(1)
            import shutil
            shutil.copy2(temp_path, final_output)
        else:
            return f"Could not locate downloaded file: {result}"
    else:
        raise ValueError("method must be 'planetary', 'gee', or 'auto'")

    # Verify the raster has proper categorical values
    try:
        with rasterio.open(final_output) as src:
            data = src.read(1)
            unique_values = np.unique(data[~np.isnan(data)])
            
            # Check if values are integers (categorical)
            if not np.all(np.equal(np.mod(unique_values, 1), 0)):
                print(f"Warning: LULC data contains non-integer values. Converting to categories...")
                
                # Map to integer categories
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                flat_data = data[~np.isnan(data)].flatten()
                le.fit(flat_data)
                transformed = le.transform(flat_data).reshape(data.shape)
                
                # Save with integer values
                with rasterio.open(final_output, 'w', **src.profile) as dst:
                    dst.write(transformed.astype(np.int16), 1)
                
                print(f"Converted LULC to {len(le.classes_)} integer categories")
    except Exception as e:
        print(f"Warning: Could not verify LULC data format: {e}")
    
    return final_output



@tool
def complete_modeling_workflow_with_lulc(
    species: str = "Apis mellifera",
    country: str = "Kenya",
    include_lulc: bool = True,
    lulc_year: int = 2021,
    n_occurrences: int = 300,
    output_dir: str = "./data/outputs"
) -> Dict[str, Any]:
    """
    Complete workflow from data download to modeling with proper LULC integration.
    """
    import time
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    steps_log = []
    results = {}
    
    try:
        # Step 1: Download environmental data
        steps_log.append(f"{timestamp} - Step 1: Downloading environmental data")
        
        # Bioclimatic data
        steps_log.append("  Downloading bioclimatic data...")
        bioclim_result = download_worldclim_temp.invoke({
            "period": "bioclim",
            "resolution": "10m",
            "out_dir": "./data/environmental"
        })
        steps_log.append(f"    Downloaded bioclim files: {len(bioclim_result)} files")
        
        # Elevation data
        steps_log.append("  Downloading elevation data...")
        elev_result = download_worldclim_temp.invoke({
            "period": "elevation",
            "resolution": "10m",
            "out_dir": "./data/environmental"
        })
        steps_log.append("    Downloaded elevation data")
        
        # LULC data (if requested)
        if include_lulc:
            steps_log.append("  Downloading LULC data...")
            lulc_result = download_lulc_for_modeling.invoke({
                "country_name": country,
                "resolution": "10m",
                "year": lulc_year,
                "output_dir": "./data/environmental",
                "method": "planetary"
            })
            
            if "Error" in lulc_result or "failed" in lulc_result.lower():
                steps_log.append(f"    WARNING: {lulc_result}")
                steps_log.append("    Continuing without LULC data...")
                include_lulc = False  # Disable LULC for rest of workflow
            else:
                steps_log.append(f"    Downloaded LULC data: {lulc_result}")
        else:
            steps_log.append("  Skipping LULC download")
        
        # Step 2: Download occurrence data
        steps_log.append(f"\n{timestamp} - Step 2: Downloading occurrence data")
        country_code = country[:2].upper() if len(country) > 2 else country
        gbif_result = download_gbif_occurrences.invoke({
            "species": species,
            "country": country_code,
            "limit": n_occurrences,
            "output_dir": "./data/gbif_data"
        })
        steps_log.append(f"  Downloaded {gbif_result['records']} records")
        results["gbif_data"] = gbif_result["csv_path"]
        
        # Step 3: Extract environmental values
        steps_log.append(f"\n{timestamp} - Step 3: Extracting environmental values")
        
        # Define predictors based on what was downloaded
        bio_predictors = list(range(1, 20))  # All bioclim variables
        
        predictors = [f"10m_bio_{i}" for i in bio_predictors] + ["10m_elev"]
        if include_lulc:
            predictors.append("10m_lulc")
        
        extracted_csv = extract_environmental_values.invoke({
            "occurrence_csv_path": gbif_result["csv_path"],
            "environmental_folder": "./data/environmental",
            "study_area": [country],
            "predictors": predictors,
            "resolution": "10m",
            "output_csv_path": os.path.join(output_dir, f"{species.replace(' ', '_')}_with_env.csv")
        })
        
        if "Error" in extracted_csv:
            raise ValueError(extracted_csv)
            
        steps_log.append(f"  Extracted environmental values to: {extracted_csv}")
        results["extracted_data"] = extracted_csv
        
        # Step 4: Analyze collinearity
        steps_log.append(f"\n{timestamp} - Step 4: Analyzing predictor collinearity")
        
        collinearity_result = analyze_predictor_colinearity.invoke({
            "occurrence_data_path": extracted_csv,
            "predictors": predictors,
            "threshold": 0.8
        })
        steps_log.append("  Collinearity analysis complete")
        
        # Step 5: Run ecological niche model
        steps_log.append(f"\n{timestamp} - Step 5: Running ecological niche model")
        
        model_output = os.path.join(output_dir, f"{species.replace(' ', '_')}_suitability.tif")
        
        # Use the updated ecological niche model function
        try:
            # Prepare bio_predictors list (you might want to adjust based on collinearity results)
            # For now, use all 19 bioclim variables
            bio_predictors = list(range(1, 20))
            
            model_result = run_ecological_niche_model.invoke({
                "species_name": species,
                "occurrence_data_path": extracted_csv,
                "environmental_data_path": "./data/environmental",
                "output_raster_path": model_output,
                "bio_predictors": bio_predictors,
                "elevation": True,
                "resolution": "10m",
                "study_area": [country]
            })
            
            if isinstance(model_result, str) and os.path.exists(model_result):
                steps_log.append(f"  Model successful: {model_output}")
                results["model_output"] = model_output
            else:
                steps_log.append(f"  Model returned: {model_result}")
                results["model_result"] = model_result
                
        except Exception as e:
            steps_log.append(f"  Model failed: {str(e)}")
            results["model_error"] = str(e)
        
        # Final summary
        steps_log.append(f"\n{'='*60}")
        steps_log.append("WORKFLOW COMPLETE")
        steps_log.append(f"{'='*60}")
        
        results["success"] = True
        results["log"] = steps_log
        results["timestamp"] = timestamp
        
        # Save workflow log
        log_file = os.path.join(output_dir, f"workflow_log_{timestamp}.txt")
        with open(log_file, 'w') as f:
            f.write("\n".join(steps_log))
        
        results["log_file"] = log_file
        
    except Exception as e:
        steps_log.append(f"\nERROR: {str(e)}")
        import traceback
        steps_log.append(f"Traceback: {traceback.format_exc()}")
        
        results = {
            "success": False,
            "error": str(e),
            "log": steps_log,
            "timestamp": timestamp
        }
    
    return results



# tools integration
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# utilities
# def display_grAIzaSyC6eg0_XUHv5NsDl8REetPzwWsfbUJaUEkaph(graph: StateGraph):
#     try:
#         display(Image(graph.get_graph().draw_mermaid_png()))
#     except Exception:
#         # This requires some extra dependencies and is optional
#         pass

# tools
websearch_tool = TavilySearch(max_results=2)


# tools = [
#     websearch_tool, 
#     download_worldclim_temp, 
#     run_suitability_model, 
#     run_ecological_niche_model,
#     download_gbif_occurrences,
#     download_shapefile
# ]

tools = [
    websearch_tool, 
    download_worldclim_temp,
    download_lulc_for_modeling, 
    complete_modeling_workflow_with_lulc,
    download_gbif_occurrences,
    download_shapefile,
    get_bounding_box_for_country,
    extract_environmental_values,
    analyze_predictor_colinearity
]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# Nodes
llm_with_tools = llm.bind_tools(tools)

# for chat node
def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Edges
# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)

# add memory:
memory = InMemorySaver()

# build graph
graph_builder.add_node("chatbot", chatbot_node)
# graph_builder.add_node("wordclim_agent", wordclim_agent_node)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge(START, "wordclim_agent")
# graph_builder.add_edge("wordclim_agent", END)
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)
# display_graph(graph)


def format_message(msg):
    return f"{msg.type.capitalize()}: {msg.content}"


def ask_question(question: str, config: dict = {"configurable": {"thread_id": "1"}}):
    events = graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="values",
    )
    ans = "\n".join(
        f"{e['messages'][-1].content}"
        for e in events
        if e["messages"][-1].type == "ai"
    )
    # for event in events:
    #     msg = event["messages"][-1]  # last message
    #     role = msg.get("role", "unknown").capitalize()
    #     content = msg.get("content", "")
    #     ans += f"\n{role}: {content}"
    #     # ans += "\n" + str(event["messages"][-1]) #.pretty_print()
    return ans



if __name__ == "__main__":
    # q1 = """
    #     For the following instructions, use tools outputs to get full file paths
    #     save the worldclim data in ./data/environmental
    #     save elevation data in ./data/environmental
    #     save GBIF data in ./data/gbif_data/
    #     save final results in ./data/outputs

    #     Download bioclim from worldclim with 10m resolution
    #     Download Elevation data with 10m resolution
    #     Download the Kenya shapefiles
    #     Download 100 records of Anopheles gambiae occurrences in Kenya from GBIF
    #     And run an ecological niche model using the downloaded elevation and occurrence data,
    #     considering bioclimatic variables 1 to 5.

    #     """

    # user_input = q1
    # ask_question(user_input)

    simple_query = """Run the complete ecological niche modeling workflow for Apis mellifera in Kenya.
    Use the complete_modeling_workflow_with_lulc tool with default settings."""
    ask_question(simple_query)