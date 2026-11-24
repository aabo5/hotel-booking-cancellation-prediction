"""
data_loader.py - Data Loading and Cleaning Utilities

This module provides simple, beginner-friendly functions to:
1. Load the raw hotel booking dataset from data/raw/
2. Validate dataset dimensions and expected columns
3. Perform basic cleaning of real-world anomalies (ghost bookings, ADR pricing errors)
"""

from pathlib import Path
import pandas as pd
import numpy as np


def get_raw_data_path() -> Path:
    """Returns the expected Path to hotel_bookings.csv."""
    # Check standard relative paths depending on where the script/notebook is run
    candidates = [
        Path("data/raw/hotel_bookings.csv"),
        Path("../data/raw/hotel_bookings.csv"),
        Path("../../data/raw/hotel_bookings.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate 'hotel_bookings.csv' in data/raw/")


def load_raw_data() -> pd.DataFrame:
    """
    Loads the raw hotel booking demand dataset into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: Raw dataset with 119,390 rows and 32 columns.
    """
    path = get_raw_data_path()
    df = pd.read_csv(path)
    print(f"Loaded raw dataset from: {path}")
    print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean_anomalies(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleans known domain anomalies from the dataset:
    1. Removes 'ghost bookings' where adults + children + babies == 0 (no guests)
    2. Filters out the extreme ADR pricing typo ($5,400) and negative room charges ($ -6.38)
    
    Parameters:
        df: Raw or partially processed DataFrame
        verbose: Whether to print summary of removed records
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # 1. Filter out ghost bookings (reservations with zero total guests)
    has_guests = (df['adults'] + df['children'].fillna(0) + df['babies']) > 0
    df_clean = df[has_guests].copy()
    ghost_removed = initial_rows - len(df_clean)
    
    # 2. Filter ADR anomalies (valid rates between $0 and $1,000 per night)
    valid_adr = (df_clean['adr'] >= 0) & (df_clean['adr'] < 1000)
    df_clean = df_clean[valid_adr].copy()
    adr_removed = (initial_rows - ghost_removed) - len(df_clean)
    
    if verbose:
        print(f"Initial records: {initial_rows:,}")
        print(f"Removed ghost bookings (0 guests): {ghost_removed:,}")
        print(f"Removed ADR anomalies (negative or > $1,000): {adr_removed:,}")
        print(f"Cleaned records remaining: {len(df_clean):,} ({(len(df_clean)/initial_rows)*100:.2f}%)")
        
    return df_clean
