import numpy as np


def CF_process_wind(x):
    """Piecewise wind capacity factor curve (based on cut-in/rated/cut-out thresholds)."""
    conditions = [(x < 3) | (x > 25), (x < 5), (x < 12), (x <= 25)]
    CFs = [
        0,
        125 * (x + 1e-7 - 3) / 2500,                     # ramp up (3–5 m/s)
        (250 + 2250 / 7 * (x - 5)) / 2500,               # ramp to rated (5–12 m/s)
        1                                                # rated (12–25 m/s)
    ]
    return np.select(conditions, CFs, default=np.nan)


def pv_power(T, S, wind10):
    """
    Hourly PV capacity factor proxy.

    Args:
        T: ambient temperature (°C)
        S: solar radiation (W/m²)
        wind10: wind speed at 10 m (m/s)

    Returns:
        PV capacity factor (dimensionless), accounting for temperature derating and system losses.
    """
    # Module temperature proxy (depends on ambient temperature, irradiance, and wind cooling)
    T_p = 4.3 + 0.943 * T + 0.028 * S / 3600 - 1.528 * wind10

    # Temperature efficiency modifier:
    # above 25°C, efficiency decreases by ~0.5% per additional °C
    TEM = (1 - 0.005 * (T_p - 25))

    # System efficiency (e.g., inverter + wiring losses)
    SYS = 0.85

    return TEM * SYS


import xarray as xr


def scale_alpha():
    """
    Compute wind speed shear exponent alpha for scaling wind speeds between heights.

    Alpha is computed from 10 m and 100 m wind speeds:
        alpha = log(wind100 / wind10) / log(100 / 10) = log(wind100 / wind10) / log(10)

    This can be used for vertical wind speed adjustment.
    """
    # Load 2020 meteorological data
    rawdata = xr.open_dataset("../RawClimateData/2020.nc")

    # Loop through 12 months
    for i in range(12):
        month = i + 1

        # Select data for the given month in 2020
        time_mask = (
            (rawdata["valid_time"].dt.month == month)
            & (rawdata["valid_time"].dt.year == 2020)
        )

        # Monthly means of wind components
        u100_temp = rawdata["u100"].where(time_mask, drop=True).mean(dim="valid_time")
        v100_temp = rawdata["v100"].where(time_mask, drop=True).mean(dim="valid_time")
        u10_temp  = rawdata["u10"].where(time_mask, drop=True).mean(dim="valid_time")
        v10_temp  = rawdata["v10"].where(time_mask, drop=True).mean(dim="valid_time")

        # Wind speed magnitude and alpha
        windspeed100 = np.sqrt(u100_temp**2 + v100_temp**2)
        windspeed10  = np.sqrt(u10_temp**2 + v10_temp**2)
        alpha = np.log(windspeed100 / windspeed10) / np.log(10)

        # Save one file per month
        alpha.to_dataset(name="alpha").to_netcdf(f"wind_scale_alpha/alpha_{month}.nc")
