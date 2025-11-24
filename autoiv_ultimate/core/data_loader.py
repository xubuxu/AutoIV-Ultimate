import os
import json
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

class DataLoader:
    """Smart data ingestion and cleaning.
    This class will later combine logic from both original projects.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Default thresholds – can be overridden via config
        self.eff_min = self.config.get('eff_min', 0.0)
        self.ff_range = self.config.get('ff_range', (0.0, 100.0))
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.outlier_method = self.config.get('outlier_method', 'iqr')

    def _detect_delimiter(self, sample: str) -> str:
        """Detect CSV delimiter (comma, semicolon, tab)."""
        for delim in [',', ';', '\t']:
            if delim in sample:
                return delim
        return ','

    def _read_csv(self, path: str) -> pd.DataFrame:
        """Read a CSV file with smart encoding and delimiter detection."""
        # Try common encodings
        for enc in ['utf-8', 'gbk', 'latin1']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    sample = f.read(1024)
                delim = self._detect_delimiter(sample)
                df = pd.read_csv(path, delimiter=delim, encoding=enc)
                return df
            except Exception:
                continue
        raise ValueError(f"Unable to read file {path} with supported encodings.")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning: threshold filters, duplicate removal, outlier removal."""
        # Threshold filters (example columns)
        if 'Eff' in df.columns:
            df = df[df['Eff'] >= self.eff_min]
        if 'FF' in df.columns:
            ff_min, ff_max = self.ff_range
            df = df[(df['FF'] >= ff_min) & (df['FF'] <= ff_max)]
        # Duplicate removal
        if self.remove_duplicates:
            df = df.drop_duplicates()
        # Outlier removal (IQR placeholder)
        if self.outlier_method == 'iqr':
            for col in ['Eff', 'Voc', 'Jsc', 'FF']:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    def load_dataset(self, folder_path: str) -> pd.DataFrame:
        """Load and clean all CSV files in a folder.
        Returns a concatenated DataFrame.
        """
        all_frames: List[pd.DataFrame] = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    full_path = os.path.join(root, file)
                    try:
                        df = self._read_csv(full_path)
                        df = self._clean_dataframe(df)
                        df['source_file'] = full_path
                        all_frames.append(df)
                    except Exception as e:
                        print(f"Failed to load {full_path}: {e}")
        if all_frames:
            return pd.concat(all_frames, ignore_index=True)
        return pd.DataFrame()
