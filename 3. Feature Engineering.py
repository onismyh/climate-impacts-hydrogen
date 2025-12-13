import pandas as pd

###################### Attribute model feature engineering ######################

feature_df = pd.read_csv("feature_matrix_cst.csv")
feature_df.rename(columns={"lon": "longitude", "lat": "latitude"}, inplace=True)
feature_df.dropna(inplace=True)
feature_df.head()


def build_minimal_features(df: pd.DataFrame):
    """
    Build a compact, interpretable feature set:
    - Tech cost deltas (direct terms)
    - Wind/Solar/Temperature summary stats (levels, variability, tails)
    - A small number of interactions with clear meaning
    """
    out = pd.DataFrame(index=df.index)

    # ---- Tech-cost deltas (direct terms) ----
    out["d_cost_pv"]      = df["PV"]
    out["d_cost_wind"]    = df["Wind"]
    out["d_cost_storage"] = df["Li-ion storage"]
    out["d_cost_elec"]    = df["Electrolyzer"]

    # ---- Wind (summary stats as physical proxies) ----
    # Central tendency / typical level
    out["wind_level"]     = df["wind100_q50"] if "wind100_q50" in df else df["wind100_avg"]
    # Variability (use var; alternatively could use IQR = q75-q25)
    out["wind_var"]       = df["wind100_var"]
    # Peakiness (peak minus mean)
    out["wind_peakiness"] = df["wind100_max"] - df["wind100_avg"]
    # Firmness / low-end supply (q10)
    out["wind_firmness"]  = df["wind100_q10"]
    # Optional spread (IQR)
    out["wind_spread"]    = df["wind100_q75"] - df["wind100_q25"]

    # ---- Solar (summary stats) ----
    out["ssrd_level"]     = df["ssrd_q50"] if "ssrd_q50" in df else df["ssrd_avg"]
    out["ssrd_var"]       = df["ssrd_var"]
    out["ssrd_peakiness"] = df["ssrd_max"] - df["ssrd_avg"]
    out["ssrd_firmness"]  = df["ssrd_q10"]
    out["ssrd_spread"]    = df["ssrd_q75"] - df["ssrd_q25"]

    # ---- Temperature (summary stats) ----
    out["t_hot"]          = df["t2m_q90"]              # heat stress proxy (warming -> higher)
    out["t_cold"]         = -df["t2m_q10"]             # cold stress proxy (colder -> larger positive)
    out["t_spread"]       = df["t2m_q75"] - df["t2m_q25"]
    out["t_var"]          = df["t2m_var"]

    # ---- Interactions (minimal set with strong interpretability) ----
    out["pv_cost_x_level"]       = df["PV"] * out["ssrd_level"]
    out["wind_cost_x_level"]     = df["Wind"] * out["wind_level"]
    out["storage_cost_x_variab"] = df["Li-ion storage"] * (out["wind_var"] + out["ssrd_var"])
    out["elec_cost_x_firmness"]  = df["Electrolyzer"] * (out["wind_firmness"] + out["ssrd_firmness"])

    # ---- Keep a minimal column subset (lean & interpretable) ----
    minimal_cols = [
        # Direct costs
        "d_cost_pv", "d_cost_wind", "d_cost_storage", "d_cost_elec",
        # Wind (level/variability/shape/low-tail)
        "wind_level", "wind_var", "wind_peakiness", "wind_firmness",
        # Solar (level/variability/shape/low-tail)
        "ssrd_level", "ssrd_var", "ssrd_peakiness", "ssrd_firmness",
        # Temperature (hot/cold/spread)
        "t_hot", "t_cold", "t_spread",
        # Interactions
        "pv_cost_x_level", "wind_cost_x_level", "storage_cost_x_variab", "elec_cost_x_firmness",
    ]

    out_min = out[minimal_cols].copy()
    out_min["target"] = df["target"].copy()

    # Optional metadata columns (uncomment if you want to keep them)
    # out_min["longitude"] = df["longitude"].copy()
    # out_min["latitude"] = df["latitude"].copy()
    # out_min["climate_model"] = df["climate_model"].copy()

    # ---- Feature grouping labels (useful for grouped SHAP and plotting) ----
    feat2group = {
        # Costs
        "d_cost_pv": "Tech-cost",
        "d_cost_wind": "Tech-cost",
        "d_cost_storage": "Tech-cost",
        "d_cost_elec": "Tech-cost",
        # Wind
        "wind_level": "Wind/mean",
        "wind_var": "Wind/variability",
        "wind_peakiness": "Wind/extreme-shape",
        "wind_firmness": "Wind/firmness",
        # Solar
        "ssrd_level": "Solar/mean",
        "ssrd_var": "Solar/variability",
        "ssrd_peakiness": "Solar/extreme-shape",
        "ssrd_firmness": "Solar/firmness",
        # Temperature
        "t_hot": "Temp/hot",
        "t_cold": "Temp/cold",
        "t_spread": "Temp/spread",
        # Interactions
        "pv_cost_x_level": "Interaction",
        "wind_cost_x_level": "Interaction",
        "storage_cost_x_variab": "Interaction",
        "elec_cost_x_firmness": "Interaction",
    }

    return out_min, feat2group, minimal_cols


feature_df_process, feat2group, minimal_cols = build_minimal_features(feature_df)
print(feature_df_process.shape)
feature_df_process.head()
