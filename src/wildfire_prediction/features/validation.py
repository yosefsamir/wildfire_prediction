import pandas as pd
from typing import Dict, Optional

def validate_dataframe(df: pd.DataFrame, schema: Dict, description: str) -> Optional[str]:
    """Validates DataFrame against schema and returns error message if invalid"""
    missing_cols = [col for col in schema.keys() if col not in df.columns]
    if missing_cols:
        return f"Missing columns {missing_cols} in {description}"
    return None

def check_missing_values(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    """Checks if missing values exceed threshold"""
    missing_ratio = df.isnull().mean().max()
    return missing_ratio <= threshold

class WildfireSchema:
    BASIC_SCHEMA = {
        "latitude": "float64",
        "longitude": "float64", 
        "brightness": "float64",
        "acq_date": "datetime64[ns]"
    }