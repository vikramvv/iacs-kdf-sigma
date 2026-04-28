import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Reference conductivity for 100% IACS (pure copper at 20°C)
SIGMA_CU_MS_M = 58.0  # MS/m

def conductivity_to_iacs(sigma_ms_m: float) -> float:
    """Convert conductivity in MS/m to %IACS."""
    return (sigma_ms_m / SIGMA_CU_MS_M) * 100.0

def parse_eddy_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric conductivity values from eddy current CSV format.

    Expects 'Value' column with format: 'RC[current_value]=X.XX MS/m'
    Adds 'Conductivity_MS_m' column with extracted float values.
    """
    if 'Value' not in df.columns:
        raise ValueError("CSV must contain 'Value' column")

    # Extract numeric value from "RC[current_value]=X.XX MS/m"
    pattern = re.compile(r"RC\[current_value\]=([\d\.]+)")
    df = df.copy()
    df['Conductivity_MS_m'] = df['Value'].astype(str).str.extract(pattern).astype(float)

    # Drop rows where extraction failed
    df = df.dropna(subset=['Conductivity_MS_m'])

    return df

def analyze_by_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eddy current measurements by (Row, Col) grid positions.

    Returns DataFrame with columns: Row, Col, mean_conductivity, std_conductivity, count
    """
    if 'Row' not in df.columns or 'Col' not in df.columns:
        # Try to extract from 'Cell' column if present (e.g., "R2C1")
        if 'Cell' in df.columns:
            df = df.copy()
            df['Row'] = df['Cell'].str.extract(r"R(\d+)").astype(int)
            df['Col'] = df['Cell'].str.extract(r"C(\d+)").astype(int)
        else:
            raise ValueError("Data must contain Row/Col columns or Cell column for extraction")

    # Group by Row, Col and compute statistics
    stats = df.groupby(['Row', 'Col'])['Conductivity_MS_m'].agg(
        mean_conductivity='mean',
        std_conductivity='std',
        count='count'
    ).reset_index()

    # Fill NaN std with 0 for single measurements
    stats['std_conductivity'] = stats['std_conductivity'].fillna(0.0)

    return stats

def analyze_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Extract left-to-right conductivity trend across columns.

    Assumes Col represents position (left-to-right).
    Returns DataFrame with columns: Col, mean_conductivity, std_conductivity, count
    """
    if 'Col' not in df.columns:
        if 'Cell' in df.columns:
            df = df.copy()
            df['Col'] = df['Cell'].str.extract(r"C(\d+)").astype(int)
        else:
            raise ValueError("Data must contain Col column or Cell column for extraction")

    # Group by Col (position) and compute statistics
    trend = df.groupby('Col')['Conductivity_MS_m'].agg(
        mean_conductivity='mean',
        std_conductivity='std',
        count='count'
    ).reset_index()

    trend['std_conductivity'] = trend['std_conductivity'].fillna(0.0)

    return trend

def parse_eddy_filename(filename: str) -> Optional[Dict]:
    """Parse eddy current filename to extract metadata.

    Expected format: {sample}-{pos}{number}-{layer}_{timestamp}.csv
    e.g., cvd2-pos4-top_20260424_150912.csv

    Returns dict with keys: sample_name, location, layer, timestamp
    """
    # Pattern: name-posN-layer_timestamp.csv
    pattern = re.compile(r'^([a-z0-9]+)-pos(\d+)-([a-z]+)_(\d{8}_\d{6})\.csv$', re.IGNORECASE)
    match = pattern.match(filename)
    if not match:
        return None

    name, pos_num, layer, timestamp = match.groups()
    return {
        'sample_name': name.lower(),
        'location': f'pos{pos_num}',
        'layer': layer.lower(),
        'timestamp': timestamp,
        'key': f"{name.lower()}-{f'pos{pos_num}'}",
        'group_key': f"{name.lower()}-{f'pos{pos_num}'}-{layer.lower()}",
    }

def load_eddy_files(uploaded_files) -> Dict[str, pd.DataFrame]:
    """Load and validate multiple eddy current CSV files.

    Returns dict: filename -> parsed DataFrame with Conductivity_MS_m column
    """
    eddy_data = {}

    for uf in uploaded_files:
        try:
            df_raw = pd.read_csv(uf)
            df_parsed = parse_eddy_csv(df_raw)
            if not df_parsed.empty:
                eddy_data[uf.name] = df_parsed
            else:
                print(f"Warning: No valid data in {uf.name}")
        except Exception as e:
            print(f"Error loading {uf.name}: {e}")

    return eddy_data

def process_eddy_batch(eddy_data: Dict[str, pd.DataFrame],
                      analysis_mode: str = 'grid') -> Dict[str, pd.DataFrame]:
    """Process batch of eddy current data files.

    analysis_mode: 'grid' for Row/Col statistics, 'trend' for position trends

    Returns dict: filename -> analysis DataFrame
    """
    results = {}

    for filename, df in eddy_data.items():
        try:
            if analysis_mode == 'grid':
                results[filename] = analyze_by_grid(df)
            elif analysis_mode == 'trend':
                results[filename] = analyze_by_position(df)
            else:
                raise ValueError(f"Unknown analysis_mode: {analysis_mode}")
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")

    return results