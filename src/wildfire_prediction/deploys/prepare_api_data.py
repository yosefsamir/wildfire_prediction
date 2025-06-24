import datetime
import requests
import pandas as pd
import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from wildfire_prediction.features import (
    engineer_ca_features
)


def fetch_weather_features(lat, lon, date) -> dict:
    """
    Fetches the last 7 days of daily weather features for a given location.
    Returns a dict with 'dates', 'tmax', 'vpdmax', and 'ppt' lists.
    """
    # parse date string
    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=6)).strftime("%Y-%m-%d")
    end   = date

    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "daily":      "temperature_2m_max,vapor_pressure_deficit_max,precipitation_sum",
            "timezone":   "auto"
        }
    )
    data = resp.json()

    try:
        dates = data["daily"]["time"]
        tmaxs = data["daily"]["temperature_2m_max"]
        vpdms = data["daily"]["vapor_pressure_deficit_max"]
        ppts  = data["daily"]["precipitation_sum"]
        # ...
        return {
            "dates": dates,
            "tmax":  tmaxs,
            "vpdmax": vpdms,
            "ppt":   ppts,
            "lat":   lat,
            "lon":   lon
        }
    except Exception as e:
        raise ValueError(f"Failed to extract 7-day weather data: {e}")


def prepare_data(data: dict) -> pd.DataFrame:
    """
    data should be a dict with
    - 'dates':  list of YYYY‑MM‑DD strings (len ≤ 7)
    - 'tmax':   list of floats, same length
    - 'vpdmax': list of floats
    - 'ppt':    list of floats
    - 'lat':    float
    - 'lon':    float

    Returns a one‑row DataFrame (today’s engineered features).
    """
    # 1) Create a multi‑row raw DataFrame
    df = pd.DataFrame({
        'date':     pd.to_datetime(data['dates']),
        'tmax':     data['tmax'],
        'vbdmax':   data['vpdmax'],
        'ppt':      data['ppt'],
    })
    # attach the constant spatial coords
    df['latitude']  = data['lat']
    df['longitude'] = data['lon']



# your original DF
    f = pd.DataFrame({
        'date':   pd.to_datetime(data['dates']),
        'tmax':   data['tmax'],
        'vbdmax': data['vpdmax'],
        'ppt':    data['ppt'],
    })
    f['latitude']  = data['lat']
    f['longitude'] = data['lon']

    # sort & index by date
    f = f.sort_values('date').set_index('date')

    # rolling 7‑day means
    f['tmax_7day_mean'] = f['tmax'].rolling(7, min_periods=1).mean()
    f['ppt_7day_mean']  = f['ppt'].rolling(7, min_periods=1).mean()
    f['vbd_7day_mean']  = f['vbdmax'].rolling(7, min_periods=1).mean()

    # basic thresholds
    TH_TMAX = 30    # °C threshold for “hot”
    TH_PPT  = 1     # mm threshold for “dry”
    f['high_temp_day'] = (f['tmax'] > TH_TMAX).astype(int)
    f['low_rain_day']  = (f['ppt'] < TH_PPT).astype(int)
    f['hot_dry_day']   = ((f['high_temp_day']==1) & (f['low_rain_day']==1)).astype(int)

    # hot-dry index (normalized product)
    f['hot_dry_index'] = (f['tmax']/f['tmax'].max()) * ((f['ppt'].max() - f['ppt'])/f['ppt'].max())

    # VPD extreme & anomaly
    VPD_EXT = 2.5  # kPa
    f['vpd_extreme'] = (f['vbdmax'] > VPD_EXT).astype(int)
    f['vpd_anomaly'] = f['vbdmax'] - f['vbd_7day_mean']

    # categorical risk
    f['drought_category']  = pd.cut(f['ppt_7day_mean'],
                                    bins=[-1,1,5,10,20,np.inf],
                                    labels=[0,1,2,3,4]).astype(int)
    f['vpd_risk_category'] = pd.cut(f['vpd_anomaly'],
                                    bins=[-np.inf,-1,1,np.inf],
                                    labels=[0,1,2]).astype(int)

    # seasonality flags & sin/cos transforms
    today = f.index.max()
    m     = today.month
    w     = today.isocalendar().week

    f['is_fire_season']      = int(m in [5,6,7,8,9,10])
    f['is_santa_ana_season'] = int(m in [10,11,12,1,2])
    f['season'] = pd.cut([m], bins=[0,3,6,9,12],
                        labels=[1,2,3,4])[0]
    f['week_sin']  = np.sin(2*np.pi * w/52)
    f['week_cos']  = np.cos(2*np.pi * w/52)
    f['month_sin'] = np.sin(2*np.pi * m/12)
    f['month_cos'] = np.cos(2*np.pi * m/12)

    # composite indices
    f['fire_weather_index'] = (f['tmax'] * f['vbdmax'] * f['hot_dry_index'])**(1/3)
    dwi = ((f['vpd_anomaly'] + 3)/6) * (1 - f['ppt_7day_mean']/f['ppt_7day_mean'].max())
    f['drought_weather_index'] = (dwi - dwi.min())/(dwi.max() - dwi.min() + 1e-6)
    f['fire_risk_index']      = (f['fire_weather_index'] + f['drought_weather_index'])/2

    # select exactly your requested columns for today
    cols = [
        'vbdmax','tmax','ppt',
        'hot_dry_index','high_temp_day','low_rain_day','hot_dry_day',
        'drought_category','vpd_extreme','vpd_anomaly','vpd_risk_category',
        'is_fire_season','is_santa_ana_season','season',
        'week_sin','week_cos','month_sin','month_cos',
        'tmax_7day_mean','ppt_7day_mean','vbd_7day_mean',
        'fire_weather_index','drought_weather_index','fire_risk_index',
        'latitude','longitude'
    ]

    feat_today = f.loc[[today], cols].reset_index(drop=True)
    print(feat_today)

    # 2) Run the full CA feature pipeline over all 7 rows
    
    return feat_today
