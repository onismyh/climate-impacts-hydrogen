import xarray as xr
import numpy as np
import pandas as pd
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import lightgbm as lgb
import logging
import os
import pickle
import xgboost as xgb


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


len_years = [248.0, 224.0, 248.0, 240.0, 248.0, 240.0, 248.0, 248.0, 240.0, 248.0, 240.0, 248.0]


def get_invest(cost_dict):
    """
    Parse investment cost variables from a dictionary.

    Args:
        cost_dict (dict): Must contain keys:
            "Wind", "PV", "Li-ion storage", "Electrolyzer"

    Returns:
        tuple: (wind_cost, pv_cost, battery_cost, electrolyzer_cost, flex)
    """
    try:
        wind_cost = cost_dict["Wind"]
        pv_cost = cost_dict["PV"]
        battery_cost = cost_dict["Li-ion storage"]
        electrolyzer_cost = cost_dict["Electrolyzer"]
        flex = 1
        return wind_cost, pv_cost, battery_cost, electrolyzer_cost, flex
    except KeyError as e:
        raise KeyError(f"Missing required key: {e}")


def auxiliary_layer(BASE_PATH):
    """Load auxiliary layers (water mask + wind alpha layers) and align them to rsds grid."""
    logger.info("Loading auxiliary layers...")

    rsds = xr.open_dataset(f"{BASE_PATH}\\used\\ssp245_2050_rsds.nc")
    rsds_temp = (rsds["rsds"] * 3600).astype(np.float32)

    # Water mask / land-availability cover (no-water cover)
    file_list = [f"../Cover_nowater/Cover_noWater_segment_{60*i}.nc" for i in range(24)]
    datasets = [xr.open_dataset(file) for file in file_list]
    wind_cover = xr.concat(datasets, dim="longitude")
    wind_cover = wind_cover.interp(latitude=rsds_temp["lat"], longitude=rsds_temp["lon"])

    # Wind speed adjustment factor layers (alpha, monthly)
    ds_alpha = [xr.open_dataset(x) for x in [f"wind_scale_alpha/alpha_{i}.nc" for i in range(1, 13)]]
    ds_alpha_temp = [ds.interp(latitude=rsds_temp["lat"], longitude=rsds_temp["lon"]) for ds in ds_alpha]

    return wind_cover, ds_alpha_temp


def process_grid_point(lat_idx, lon_idx, wind_cover, ds_alpha_temp, cost_dict):
    """Compute climate statistics + attach cost parameters for one grid cell."""
    if wind_cover["Cover_noWater"].values[lat_idx, lon_idx] == 0:
        return None

    stats = {
        "latitude": rsds_temp["lat"].values[lat_idx],
        "longitude": rsds_temp["lon"].values[lon_idx],
        "wind_cost": cost_dict.get("Wind", None),
        "pv_cost": cost_dict.get("PV", None),
        "battery_cost": cost_dict.get("Li-ion storage", None),
        "electrolyzer_cost": cost_dict.get("Electrolyzer", None),
        "Flex": 1,
    }

    # Check again (kept as-is to match your original logic)
    if wind_cover["Cover_noWater"].values[lat_idx, lon_idx] == 0:
        return None

    # Compute variables (time series)
    alpha_mon = [ds["alpha"][lat_idx, lon_idx].item() for ds in ds_alpha_temp]
    variables = {
        "wind100": np.sqrt(
            (uas_temp[:, lat_idx, lon_idx].values[:2920] * np.power(
                10, (np.repeat(alpha_mon, len_years)[:uas_temp.sizes["time"]])
            )) ** 2
            + (vas_temp[:, lat_idx, lon_idx].values[:2920] * np.power(
                10, (np.repeat(alpha_mon, len_years)[:vas_temp.sizes["time"]])
            )) ** 2
        ),
        "wind10": np.sqrt(
            uas_temp[:, lat_idx, lon_idx].values ** 2
            + vas_temp[:, lat_idx, lon_idx].values ** 2
        ),
        "ssrd": rsds_temp[:, lat_idx, lon_idx].values,
        "t2m": tas_temp[:, lat_idx, lon_idx].values - 273.15,  # K -> °C
    }

    # Summary statistics for each variable
    for var_name, var_data in variables.items():
        if var_data.size > 0:
            mean_val = var_data.mean()
            stats.update({
                f"{var_name}_avg": mean_val,
                f"{var_name}_var": var_data.var(),
                f"{var_name}_variation": (np.sqrt(var_data.var()) / mean_val) if mean_val != 0 else np.nan,
                f"{var_name}_q10": np.percentile(var_data, 10),
                f"{var_name}_q25": np.percentile(var_data, 25),
                f"{var_name}_q50": np.percentile(var_data, 50),
                f"{var_name}_q75": np.percentile(var_data, 75),
                f"{var_name}_q90": np.percentile(var_data, 90),
                f"{var_name}_max": var_data.max(),
            })

    return stats


# Function to create a 2D matrix from point predictions
def create_data_matrix(data_pred, lat_values, lon_values):
    dat_matrix = np.full((len(lat_values), len(lon_values)), np.nan)
    lat_idx_map = {lat: idx for idx, lat in enumerate(lat_values)}
    lon_idx_map = {lon: idx for idx, lon in enumerate(lon_values)}

    for _, row in data_pred.iterrows():
        lat_idx = lat_idx_map.get(row["latitude"])
        lon_idx = lon_idx_map.get(row["longitude"])
        if lat_idx is not None and lon_idx is not None:
            dat_matrix[lat_idx, lon_idx] = row["hydrogen_cost"]

    return dat_matrix


def process_grid_point_base(lat_idx, lon_idx, wind_cover, ds_alpha_temp):
    """
    Compute climate features only (no cost values), but keep placeholders
    for cost fields so the final column schema stays consistent.
    """
    stats = {
        "latitude": rsds_temp["lat"].values[lat_idx],
        "longitude": rsds_temp["lon"].values[lon_idx],
        "wind_cost": None,
        "pv_cost": None,
        "battery_cost": None,
        "electrolyzer_cost": None,
        "Flex": None,
    }

    # Water mask (skip invalid cells)
    if wind_cover["Cover_noWater"].values[lat_idx, lon_idx] == 0:
        return None

    try:
        alpha_mon = [ds["alpha"][lat_idx, lon_idx].item() for ds in ds_alpha_temp]
        variables = {
            "wind100": np.sqrt(
                (uas_temp[:, lat_idx, lon_idx].values[:2920] * np.power(
                    10, (np.repeat(alpha_mon, len_years)[:uas_temp.sizes["time"]])
                )) ** 2
                + (vas_temp[:, lat_idx, lon_idx].values[:2920] * np.power(
                    10, (np.repeat(alpha_mon, len_years)[:vas_temp.sizes["time"]])
                )) ** 2
            ),
            "wind10": np.sqrt(
                uas_temp[:, lat_idx, lon_idx].values ** 2
                + vas_temp[:, lat_idx, lon_idx].values ** 2
            ),
            "ssrd": rsds_temp[:, lat_idx, lon_idx].values,
            "t2m": tas_temp[:, lat_idx, lon_idx].values - 273.15,
        }

        for var_name, var_data in variables.items():
            if var_data.size > 0:
                mean_val = var_data.mean()
                stats.update({
                    f"{var_name}_avg": mean_val,
                    f"{var_name}_var": var_data.var(),
                    f"{var_name}_variation": (np.sqrt(var_data.var()) / mean_val) if mean_val != 0 else np.nan,
                    f"{var_name}_q10": np.percentile(var_data, 10),
                    f"{var_name}_q25": np.percentile(var_data, 25),
                    f"{var_name}_q50": np.percentile(var_data, 50),
                    f"{var_name}_q75": np.percentile(var_data, 75),
                    f"{var_name}_q90": np.percentile(var_data, 90),
                    f"{var_name}_max": var_data.max(),
                })

        return stats

    except Exception as e:
        logger.warning(f"Error processing grid point (lat_idx={lat_idx}, lon_idx={lon_idx}): {e}")
        return None


def process_grid_point_with_cost(base_stats, cost_dict):
    """Update cost parameters on top of precomputed climate statistics."""
    if base_stats is None:
        return None
    wind_cost, pv_cost, battery_cost, electrolyzer_cost, Flex = get_invest(cost_dict)
    stats = base_stats.copy()
    stats["wind_cost"] = wind_cost
    stats["pv_cost"] = pv_cost
    stats["battery_cost"] = battery_cost
    stats["electrolyzer_cost"] = electrolyzer_cost
    stats["Flex"] = Flex
    return stats


# Function to create a 2D matrix from point predictions (generic id column)
def create_data_matrix(data_pred, lat_values, lon_values, id="hydrogen_cost"):
    dat_matrix = np.full((len(lat_values), len(lon_values)), np.nan)
    lat_idx_map = {lat: idx for idx, lat in enumerate(lat_values)}
    lon_idx_map = {lon: idx for idx, lon in enumerate(lon_values)}

    for _, row in data_pred.iterrows():
        lat_idx = lat_idx_map.get(row["latitude"])
        lon_idx = lon_idx_map.get(row["longitude"])
        if lat_idx is not None and lon_idx is not None:
            dat_matrix[lat_idx, lon_idx] = row[id]

    return dat_matrix


def process_ssp(ssp, year, wind_cover, ds_alpha_temp, cost_dict, used, BASE_PATH, model_file, id="ratio"):
    logger.info(f"Processing {ssp}...{year}")

    # Load datasets
    try:
        rsds = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_rsds.nc")
        tas = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_tas.nc")
        uas = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_uas.nc")
        vas = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_vas.nc")
    except Exception as e:
        logger.error(f"Failed to load datasets for {ssp}: {e}")
        return

    # Expose to globals used by grid-point functions
    global rsds_temp, tas_temp, uas_temp, vas_temp
    rsds_temp = (rsds["rsds"] * 3600).astype("float32")
    tas_temp = tas["tas"].astype("float32")
    uas_temp = uas["uas"].astype("float32")
    vas_temp = vas["vas"].astype("float32")

    lat_values = rsds_temp["lat"].values
    lon_values = rsds_temp["lon"].values

    # Parallel grid processing via Dask
    delayed_results = [
        delayed(process_grid_point)(lat_idx, lon_idx, wind_cover, ds_alpha_temp, cost_dict)
        for lat_idx in range(len(lat_values))
        for lon_idx in range(len(lon_values))
    ]
    with ProgressBar():
        results = compute(*delayed_results)

    filtered_results = [res for res in results if res is not None]
    data_pred = pd.DataFrame(filtered_results)

    # Load model
    with open(model_file, "rb") as f:
        bst_loaded_r1 = pickle.load(f)

    # Predict (supports LightGBM Booster / XGBoost Booster / sklearn-like estimators)
    if lgb is not None and isinstance(bst_loaded_r1, (lgb.Booster, lgb.sklearn.LGBMModel)):
        pred = bst_loaded_r1.predict(data_pred)
    elif isinstance(bst_loaded_r1, xgb.Booster):
        dm_pred = xgb.DMatrix(data_pred)
        pred = bst_loaded_r1.predict(dm_pred)
    else:
        pred = bst_loaded_r1.predict(data_pred)

    data_pred[id] = pred
    dat_matrix = create_data_matrix(data_pred, lat_values, lon_values, id=id)
    return dat_matrix


def process_ssp_multi(
    ssp, year, wind_cover, ds_alpha_temp, cost_list, used, BASE_PATH, model_file, id_prefix="ratio"
):
    """
    Batch-predict multiple cost scenarios on the same climate dataset:
    - Load climate data once
    - Compute base climate features once (process_grid_point_base)
    - For each cost scenario, only swap cost parameters (process_grid_point_with_cost) and predict
    - Returns: {idx_cost: dat_matrix}
    """
    logger.info(f"Processing {ssp}...{year} for {len(cost_list)} cost scenarios")

    # 1) Load climate data (once)
    try:
        rsds = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_rsds.nc")
        tas  = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_tas.nc")
        uas  = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_uas.nc")
        vas  = xr.open_dataset(f"{BASE_PATH}\\{used}\\{ssp}_{year}_vas.nc")
    except Exception as e:
        logger.error(f"Failed to load datasets for {ssp}: {e}")
        return {}

    # 2) Convert to float32 + expose to globals for base feature computation
    global rsds_temp, tas_temp, uas_temp, vas_temp
    rsds_temp = (rsds["rsds"] * 3600).astype("float32")
    tas_temp  =  tas["tas"].astype("float32")
    uas_temp  =  uas["uas"].astype("float32")
    vas_temp  =  vas["vas"].astype("float32")

    lat_values = rsds_temp["lat"].values
    lon_values = rsds_temp["lon"].values

    # 3) Load model (once)
    with open(model_file, "rb") as f:
        bst_loaded_r1 = pickle.load(f)

    # 4) Compute base climate features for all grid points (cost fields are placeholders)
    delayed_results = [
        delayed(process_grid_point_base)(lat_idx, lon_idx, wind_cover, ds_alpha_temp)
        for lat_idx in range(len(lat_values))
        for lon_idx in range(len(lon_values))
    ]
    with ProgressBar():
        base_features = compute(*delayed_results)

    base_list = [bf for bf in base_features if bf is not None]
    if len(base_list) == 0:
        logger.warning("No valid grid points after applying the water mask.")
        return {}

    results_dict = {}

    # 5) Loop over cost scenarios: swap cost fields and predict
    for idx_cost, cost_dict in enumerate(cost_list):
        stats_with_cost = [process_grid_point_with_cost(bf, cost_dict) for bf in base_list]
        data_pred = pd.DataFrame(stats_with_cost)

        if lgb is not None and isinstance(bst_loaded_r1, (lgb.Booster, lgb.sklearn.LGBMModel)):
            pred = bst_loaded_r1.predict(data_pred)
        elif isinstance(bst_loaded_r1, xgb.Booster):
            dm_pred = xgb.DMatrix(data_pred)
            pred = bst_loaded_r1.predict(dm_pred)
        else:
            pred = bst_loaded_r1.predict(data_pred)

        colname = f"{id_prefix}_{idx_cost}"
        data_pred[colname] = pred

        dat_matrix = create_data_matrix(data_pred, lat_values, lon_values, id=colname)
        results_dict[idx_cost] = dat_matrix

    return results_dict
