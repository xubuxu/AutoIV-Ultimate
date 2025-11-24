"""
AutoIV-Ultimate Data Loader Module

Merges functionality from:
- Auto_IV_Analysis_Suite/dataloader.py (raw IV parsing, grouping)
- IV_Batch_Analyzer/data_loader.py (smart CSV parsing, header detection)
"""
import os
import re
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ================= COLUMN MAPPING =================

COLUMN_MAPPING = {
    'Eff': ['eff', 'efficiency', 'pce', 'eta'],
    'Voc': ['voc', 'uoc', 'open_circuit_voltage'],
    'Jsc': ['jsc', 'isc', 'short_circuit_current', 'j_sc'],
    'FF': ['ff', 'fill_factor'],
    'Rs': ['rs', 'rs_light', 'series_resistance'],
    'Rsh': ['rsh', 'rsh_light', 'shunt_resistance', 'rp'],
    'ScanDir': ['scandirection', 'direction', 'scan_dir'],
    'CellName': ['cellname', 'device_id', 'sample_name', 'name', 'pixel'],
    'Voltage': ['voltage', 'v', 'bias'],
    'Current': ['current', 'i', 'current_density', 'j']
}


# ================= EXCEPTIONS =================

class CSVParseError(Exception):
    """Raised when CSV file cannot be parsed."""
    pass


# ================= DATA LOADER =================

class DataLoader:
    """Unified data loader combining features from both original projects."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration dictionary with thresholds, flags, etc.
        """
        self.config = config or {}
        
        # Thresholds
        self.thresholds = self.config.get('thresholds', {})
        self.eff_min = self.thresholds.get('Eff_Min', 0.1)
        self.voc_min = self.thresholds.get('Voc_Min', 0.1)
        self.jsc_min = self.thresholds.get('Jsc_Min', 0.1)
        self.ff_min = self.thresholds.get('FF_Min', 10.0)
        self.ff_max = self.thresholds.get('FF_Max', 90.0)
        
        # Flags
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.outlier_removal = self.config.get('outlier_removal', True)
        self.scan_direction = self.config.get('scan_direction', 'Reverse')
        
        # Patterns
        self.input_patterns = self.config.get('input_patterns', [
            '*IVMeasurement*.csv', '*summary*.csv', '*result*.csv', '*data*.csv', '*.csv'
        ])
        
        logger.info(f"DataLoader initialized with scan_direction={self.scan_direction}")
    
    # ================= UTILITY METHODS =================
    
    @staticmethod
    def natural_keys(text: str) -> List[Union[int, str]]:
        """
        Helper for natural sorting (e.g., Batch_2 < Batch_10).
        
        Args:
            text: Text to convert to natural sort key
            
        Returns:
            List of integers and strings for natural sorting
        """
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]
    
    @staticmethod
    def _detect_delimiter(file_path: Path) -> str:
        """
        Auto-detect CSV delimiter using csv.Sniffer.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected delimiter character
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except Exception:
            # Fallback: check for common delimiters
            for delim in [',', ';', '\t']:
                if delim in sample:
                    return delim
            return ','
    
    @staticmethod
    def _find_header_row(file_path: Path, delimiter: str, encoding: str = 'utf-8') -> int:
        """
        Find the row containing column headers (e.g., "Voltage", "Eff", "Current").
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            encoding: File encoding
            
        Returns:
            Row index (0-based) of header row, or 0 if not found
        """
        header_keywords = ['voltage', 'current', 'v', 'i', 'efficiency', 'eff', 'voc', 'jsc', 'ff']
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for idx, line in enumerate(f):
                    if idx > 50:  # Don't search beyond first 50 rows
                        break
                    
                    line_lower = line.lower()
                    # Check if line contains at least 2 header keywords
                    keyword_count = sum(1 for kw in header_keywords if kw in line_lower)
                    
                    if keyword_count >= 2:
                        logger.debug(f"Found header row at line {idx} in {file_path.name}")
                        return idx
            
            logger.debug(f"No clear header row found in {file_path.name}, using row 0")
            return 0
            
        except Exception as e:
            logger.debug(f"Header detection failed: {e}")
            return 0
    
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Maps variable CSV headers to standard internal names.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            Dictionary mapping standard names to original column names
        """
        df.columns = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in df.columns}
        found_cols = {}

        for std_name, possible_names in COLUMN_MAPPING.items():
            for name in possible_names:
                if name in lower_cols:
                    found_cols[std_name] = lower_cols[name]
                    break
                # Fallback: partial match
                for col_lower, col_orig in lower_cols.items():
                    if name in col_lower and std_name not in found_cols:
                        found_cols[std_name] = col_orig
                        break
        return found_cols
    
    def _read_csv_smart(self, file_path: Path) -> pd.DataFrame:
        """
        Read CSV with smart parsing: auto-detect delimiter, encoding, and header row.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pandas DataFrame
            
        Raises:
            CSVParseError: If file cannot be parsed
        """
        # Try multiple encodings
        encodings_to_try = ['utf-8', 'gbk', 'latin1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                # Detect delimiter
                delimiter = self._detect_delimiter(file_path)
                
                # Find header row
                skiprows = self._find_header_row(file_path, delimiter, encoding)
                
                # Read CSV
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    skiprows=skiprows,
                    on_bad_lines='skip'
                )
                
                # Check if dataframe is valid
                if df.empty or len(df.columns) < 2:
                    continue
                
                logger.debug(f"Successfully read {file_path.name} with {encoding} encoding")
                return df
                
            except Exception as e:
                logger.debug(f"Failed to read with {encoding}: {e}")
                continue
        
        raise CSVParseError(f"Could not parse {file_path.name} with any supported encoding")
    
    # ================= CLEANING METHODS =================
    
    def _apply_thresholds(self, df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
        """Apply threshold filters to dataframe."""
        original_count = len(df)
        
        # Efficiency threshold
        if 'Eff' in col_map:
            df = df[df[col_map['Eff']] >= self.eff_min]
        
        # Voc threshold
        if 'Voc' in col_map:
            df = df[df[col_map['Voc']] >= self.voc_min]
        
        # Jsc threshold
        if 'Jsc' in col_map:
            df = df[df[col_map['Jsc']] >= self.jsc_min]
        
        # FF range
        if 'FF' in col_map:
            df = df[(df[col_map['FF']] >= self.ff_min) & (df[col_map['FF']] <= self.ff_max)]
        
        filtered_count = len(df)
        if filtered_count < original_count:
            logger.info(f"Threshold filtering: {original_count} → {filtered_count} cells")
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, col_map: Dict[str, str], multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        if not self.outlier_removal:
            return df
        
        original_count = len(df)
        
        # Apply IQR to key parameters
        for std_name in ['Eff', 'Voc', 'Jsc', 'FF']:
            if std_name in col_map:
                col = col_map[std_name]
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        filtered_count = len(df)
        if filtered_count < original_count:
            logger.info(f"Outlier removal: {original_count} → {filtered_count} cells")
        
        return df
    
    def _filter_scan_direction(self, df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
        """Filter by scan direction if column exists."""
        if 'ScanDir' not in col_map or self.scan_direction == 'All':
            return df
        
        scan_col = col_map['ScanDir']
        original_count = len(df)
        
        # Filter for specified scan direction
        df = df[df[scan_col].str.contains(self.scan_direction, case=False, na=False)]
        
        filtered_count = len(df)
        if filtered_count < original_count:
            logger.info(f"Scan direction filter ({self.scan_direction}): {original_count} → {filtered_count} cells")
        
        return df
    
    # ================= MAIN LOADING METHOD =================
    
    def load_dataset(self, folder_path: Union[str, Path]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and process all CSV files in a folder.
        
        Args:
            folder_path: Path to the data folder
            
        Returns:
            Tuple of (concatenated DataFrame, list of batch names in natural order)
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        all_frames: List[pd.DataFrame] = []
        batch_names = set()
        
        # Recursively find all CSV files
        csv_files = []
        for pattern in self.input_patterns:
            csv_files.extend(folder_path.rglob(pattern))
        
        # Remove duplicates
        csv_files = list(set(csv_files))
        
        # Sort naturally
        csv_files.sort(key=lambda p: self.natural_keys(str(p)))
        
        logger.info(f"Found {len(csv_files)} CSV files in {folder_path}")
        
        for csv_file in csv_files:
            try:
                # Read CSV with smart parsing
                df = self._read_csv_smart(csv_file)
                
                # Normalize columns
                col_map = self._normalize_columns(df)
                
                # Check for required columns
                required_cols = ['Eff', 'Voc', 'Jsc', 'FF']
                missing = [c for c in required_cols if c not in col_map]
                if missing:
                    logger.warning(f"Skipping {csv_file.name}: missing columns {missing}")
                    continue
                
                # Rename columns to standard names
                df = df.rename(columns={v: k for k, v in col_map.items()})
                
                # Add metadata
                df['source_file'] = str(csv_file)
                df['batch'] = csv_file.parent.name
                batch_names.add(csv_file.parent.name)
                
                # Apply filters
                df = self._apply_thresholds(df, {k: k for k in col_map.keys()})
                df = self._filter_scan_direction(df, {k: k for k in col_map.keys()})
                
                if not df.empty:
                    all_frames.append(df)
                    logger.debug(f"Loaded {len(df)} cells from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")
                continue
        
        if not all_frames:
            logger.warning("No valid data loaded")
            return pd.DataFrame(), []
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_frames, ignore_index=True)
        
        # Remove duplicates if enabled
        if self.remove_duplicates:
            original_count = len(combined_df)
            combined_df = combined_df.drop_duplicates()
            if len(combined_df) < original_count:
                logger.info(f"Duplicate removal: {original_count} → {len(combined_df)} cells")
        
        # Remove outliers
        col_map_combined = {k: k for k in ['Eff', 'Voc', 'Jsc', 'FF'] if k in combined_df.columns}
        combined_df = self._remove_outliers_iqr(combined_df, col_map_combined)
        
        # Get batch order (natural sorted)
        batch_order = sorted(list(batch_names), key=self.natural_keys)
        
        logger.info(f"Total loaded: {len(combined_df)} cells from {len(batch_names)} batches")
        
        return combined_df, batch_order
