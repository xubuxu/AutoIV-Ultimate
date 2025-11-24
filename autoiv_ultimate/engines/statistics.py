"""
AutoIV-Ultimate Statistics Engine Module

Ported from IV_Batch_Analyzer/statistics.py with enhancements.
Handles batch aggregation, T-tests, yield calculation, and champion cell selection.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ================= CONSTANTS =================

ANALYSIS_PARAMS = ['Eff', 'Voc', 'Jsc', 'FF', 'Rs', 'Rsh']

EFF_BINS = [0, 10, 15, 18, 20, 22, 24, 26, 30, 100]
EFF_BIN_LABELS = ['<10%', '10-15%', '15-18%', '18-20%', '20-22%', '22-24%', '24-26%', '26-30%', '>30%']


# ================= STATISTICS ENGINE =================

class StatisticsEngine:
    """Handles statistical analysis, yield calculation, and hysteresis metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize StatisticsEngine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.control_keywords = self.config.get('control_keywords', ['Ref', 'Ctrl', 'Control', 'Std', 'Baseline'])
        self.champion_criteria = self.config.get('champion_criteria', 'Max Eff')
        
        logger.info("StatisticsEngine initialized")
    
    def assign_colors(self, batch_order: List[str]) -> Dict[str, str]:
        """
        Assign consistent colors to batches.
        
        Args:
            batch_order: List of batch names in order
            
        Returns:
            Dictionary mapping batch names to colors
        """
        default_palette = [
            "#4DBBD5", "#E64B35", "#00A087", "#3C5488", "#F39B7F",
            "#8491B4", "#91D1C2", "#7E6148", "#B09C85"
        ]
        
        palette = self.config.get('colors', default_palette)
        return {batch: palette[i % len(palette)] for i, batch in enumerate(batch_order)}
    
    def compute_statistics(self, clean_df: pd.DataFrame, batch_order: List[str]) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Calculate comprehensive statistics using vectorized Pandas aggregation.
        
        Args:
            clean_df: Cleaned dataframe
            batch_order: List of batch names in order
            
        Returns:
            Tuple of (stats_df, champion_df, top_cells_df, yield_df, comparisons)
        """
        if clean_df.empty:
            logger.warning("Empty dataframe provided to compute_statistics")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
        
        logger.info("Computing statistics...")
        
        # Get available parameters
        available_params = [p for p in ANALYSIS_PARAMS if p in clean_df.columns]
        
        # Define aggregation dictionary
        agg_funcs = {param: ['mean', 'median', 'max', 'std', 'min'] for param in available_params}
        agg_funcs['batch'] = ['count']  # Count samples
        
        # Perform vectorized aggregation
        grouped = clean_df.groupby('batch')
        stats_raw = grouped.agg(agg_funcs)
        
        # Identify control batch
        control_batch = next(
            (b for b in batch_order if any(k.lower() in b.lower() for k in self.control_keywords)),
            batch_order[0] if batch_order else None
        )
        
        comparisons = {'Control': control_batch, 'Results': {}}
        control_data = clean_df[clean_df['batch'] == control_batch] if control_batch else pd.DataFrame()
        
        # Reconstruct stats dataframe with T-tests
        stats_list = []
        
        for batch in batch_order:
            if batch not in stats_raw.index:
                continue
            
            # Extract aggregated stats
            row = {
                'Batch': batch,
                'Count': int(stats_raw.loc[batch, ('batch', 'count')])
            }
            
            for param in available_params:
                mean_val = stats_raw.loc[batch, (param, 'mean')]
                std_val = stats_raw.loc[batch, (param, 'std')]
                
                row[f'{param}_Mean'] = mean_val
                row[f'{param}_Median'] = stats_raw.loc[batch, (param, 'median')]
                row[f'{param}_Max'] = stats_raw.loc[batch, (param, 'max')]
                row[f'{param}_Min'] = stats_raw.loc[batch, (param, 'min')]
                row[f'{param}_Std'] = std_val
                
                # Calculate CV% (Coefficient of Variation)
                if mean_val != 0:
                    row[f'{param}_CV%'] = (std_val / mean_val) * 100
                else:
                    row[f'{param}_CV%'] = np.nan
                
                # T-Test vs control batch
                if batch != control_batch and not control_data.empty:
                    df_b = clean_df[clean_df['batch'] == batch]
                    if len(df_b) > 1 and len(control_data) > 1 and param in df_b.columns and param in control_data.columns:
                        try:
                            a = df_b[param].dropna()
                            b = control_data[param].dropna()
                            
                            if len(a) > 1 and len(b) > 1:
                                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
                                diff = mean_val - control_data[param].mean()
                                
                                # Significance stars
                                if p_value < 0.001:
                                    sig = '***'
                                elif p_value < 0.01:
                                    sig = '**'
                                elif p_value < 0.05:
                                    sig = '*'
                                else:
                                    sig = 'ns'
                                
                                if batch not in comparisons['Results']:
                                    comparisons['Results'][batch] = {}
                                
                                comparisons['Results'][batch][param] = {
                                    'p': p_value,
                                    't': t_stat,
                                    'sig': sig,
                                    'diff': diff,
                                    'dir': "Increase" if diff > 0 else "Decrease"
                                }
                        except Exception as e:
                            logger.debug(f"T-Test failed for {batch} vs {control_batch} on {param}: {e}")
            
            stats_list.append(row)
        
        stats_df = pd.DataFrame(stats_list)
        
        # Champion cell selection
        sort_col = 'FF' if self.champion_criteria == 'Max FF' else 'Eff'
        
        if sort_col in clean_df.columns:
            champion_df = clean_df.sort_values(sort_col, ascending=False).groupby('batch', as_index=False).first()
            champion_df = champion_df.set_index('batch').reindex(batch_order).reset_index()
        else:
            champion_df = pd.DataFrame()
        
        # Top 10 cells overall
        if 'Eff' in clean_df.columns:
            top_cells_df = clean_df.sort_values('Eff', ascending=False).head(10)
        else:
            top_cells_df = pd.DataFrame()
        
        # Yield calculation
        if 'Eff' in clean_df.columns:
            # Add efficiency bins
            clean_df_copy = clean_df.copy()
            clean_df_copy['Eff_Bin'] = pd.cut(clean_df_copy['Eff'], bins=EFF_BINS, labels=EFF_BIN_LABELS)
            
            # Calculate yield percentages
            yield_raw = pd.crosstab(clean_df_copy['batch'], clean_df_copy['Eff_Bin'], normalize='index') * 100
            yield_df = yield_raw.reindex(batch_order).fillna(0).reset_index()
        else:
            yield_df = pd.DataFrame()
        
        logger.info(f"✓ Statistics computed for {len(stats_df)} batches")
        
        return stats_df, champion_df, top_cells_df, yield_df, comparisons
    
    def calculate_yield(self, df: pd.DataFrame, threshold: float = 20.0) -> float:
        """
        Calculate yield based on efficiency threshold.
        
        Args:
            df: Dataframe with Eff column
            threshold: Efficiency threshold (%)
            
        Returns:
            Yield percentage
        """
        if df.empty or 'Eff' not in df.columns:
            return 0.0
        
        total = len(df)
        passing = len(df[df['Eff'] >= threshold])
        
        yield_pct = (passing / total * 100) if total > 0 else 0.0
        
        logger.debug(f"Yield calculation: {passing}/{total} cells above {threshold}% = {yield_pct:.1f}%")
        
        return yield_pct
    
    def identify_champion_cells(self, df: pd.DataFrame, top_n: int = 5, criteria: str = 'Eff') -> pd.DataFrame:
        """
        Identify top-N champion cells based on specified criteria.
        
        Args:
            df: Dataframe with performance parameters
            top_n: Number of top cells to return
            criteria: Column to sort by (default: 'Eff')
            
        Returns:
            DataFrame with top-N cells
        """
        if df.empty or criteria not in df.columns:
            logger.warning(f"Cannot identify champion cells: empty dataframe or missing {criteria} column")
            return pd.DataFrame()
        
        champion_df = df.sort_values(criteria, ascending=False).head(top_n)
        
        logger.info(f"Identified top {len(champion_df)} champion cells by {criteria}")
        
        return champion_df
    
    def calculate_hysteresis_metrics(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate hysteresis metrics for cells with both forward and reverse scans.
        
        Important for perovskite solar cells analysis.
        
        Hysteresis Index (HI):
            HI = (PCE_reverse - PCE_forward) / PCE_reverse × 100%
        
        Categories:
            - Negligible: HI < 5%
            - Moderate: 5% <= HI < 15%
            - Significant: HI >= 15%
        
        Args:
            raw_df: Raw dataframe before filtering (must contain ScanDir column)
            
        Returns:
            DataFrame with hysteresis metrics or None if data unavailable
        """
        if 'ScanDir' not in raw_df.columns or 'CellName' not in raw_df.columns or raw_df.empty:
            logger.info("Hysteresis analysis skipped: Missing scan direction or cell name data")
            return None
        
        logger.info("Calculating hysteresis metrics...")
        
        # Separate forward and reverse scans
        df = raw_df.copy()
        df['ScanDir'] = df['ScanDir'].astype(str).str.strip().str.upper()
        
        forward_mask = df['ScanDir'].str.contains('F', case=False, na=False)
        reverse_mask = df['ScanDir'].str.contains('R', case=False, na=False)
        
        df_forward = df[forward_mask].copy()
        df_reverse = df[reverse_mask].copy()
        
        if df_forward.empty or df_reverse.empty:
            logger.warning("Hysteresis analysis skipped: Missing forward or reverse scans")
            return None
        
        # Find cells with both scans
        cells_forward = set(df_forward['CellName'].unique())
        cells_reverse = set(df_reverse['CellName'].unique())
        cells_both = cells_forward & cells_reverse
        
        if not cells_both:
            logger.warning("Hysteresis analysis skipped: No cells with both scan directions")
            return None
        
        logger.info(f"Found {len(cells_both)} cells with both forward and reverse scans")
        
        # Build hysteresis dataframe
        hysteresis_records = []
        
        for cell in cells_both:
            try:
                # Get forward and reverse data (keep best if duplicates)
                fwd = df_forward[df_forward['CellName'] == cell].sort_values('Eff', ascending=False).iloc[0]
                rev = df_reverse[df_reverse['CellName'] == cell].sort_values('Eff', ascending=False).iloc[0]
                
                # Calculate metrics
                params = ['Eff', 'Voc', 'Jsc', 'FF']
                record = {
                    'CellName': cell,
                    'Batch': fwd.get('batch', 'Unknown')
                }
                
                for param in params:
                    if param in fwd and param in rev:
                        val_fwd = fwd[param]
                        val_rev = rev[param]
                        record[f'{param}_Forward'] = val_fwd
                        record[f'{param}_Reverse'] = val_rev
                        
                        # Hysteresis Index
                        if val_rev != 0:
                            hi = ((val_rev - val_fwd) / val_rev) * 100
                            record[f'HI_{param}'] = hi
                        else:
                            record[f'HI_{param}'] = np.nan
                
                # Overall hysteresis category based on Eff
                hi_eff = record.get('HI_Eff', np.nan)
                if pd.isna(hi_eff):
                    category = 'Unknown'
                elif abs(hi_eff) < 5:
                    category = 'Negligible'
                elif abs(hi_eff) < 15:
                    category = 'Moderate'
                else:
                    category = 'Significant'
                
                record['Category'] = category
                record['Eff_Average'] = (record['Eff_Forward'] + record['Eff_Reverse']) / 2
                
                hysteresis_records.append(record)
                
            except Exception as e:
                logger.debug(f"Hysteresis calculation failed for cell {cell}: {e}")
                continue
        
        if not hysteresis_records:
            logger.warning("No hysteresis metrics could be calculated")
            return None
        
        hysteresis_df = pd.DataFrame(hysteresis_records)
        logger.info(f"✓ Hysteresis analysis complete: {len(hysteresis_df)} cells analyzed")
        
        return hysteresis_df
    
    def perform_t_test(self, group_a: pd.DataFrame, group_b: pd.DataFrame, param: str = 'Eff') -> Dict[str, float]:
        """
        Perform independent-sample T-test between two groups.
        
        Args:
            group_a: First group dataframe
            group_b: Second group dataframe
            param: Parameter to compare (default: 'Eff')
            
        Returns:
            Dictionary with t-statistic, p-value, and significance
        """
        if group_a.empty or group_b.empty or param not in group_a.columns or param not in group_b.columns:
            logger.warning(f"Cannot perform T-test: invalid data or missing {param} column")
            return {'t_stat': 0.0, 'p_value': 1.0, 'sig': 'ns'}
        
        a = group_a[param].dropna()
        b = group_b[param].dropna()
        
        if len(a) < 2 or len(b) < 2:
            logger.warning("Cannot perform T-test: insufficient data points")
            return {'t_stat': 0.0, 'p_value': 1.0, 'sig': 'ns'}
        
        try:
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            
            # Significance stars
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            logger.debug(f"T-test result: t={t_stat:.3f}, p={p_value:.4f}, sig={sig}")
            
            return {
                't_stat': t_stat,
                'p_value': p_value,
                'sig': sig,
                'mean_diff': a.mean() - b.mean()
            }
        
        except Exception as e:
            logger.error(f"T-test failed: {e}")
            return {'t_stat': 0.0, 'p_value': 1.0, 'sig': 'ns'}
    
    def aggregate_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group data by batch and compute basic statistics.
        
        Args:
            df: Dataframe with batch column
            
        Returns:
            Aggregated dataframe with statistics per batch
        """
        if df.empty or 'batch' not in df.columns:
            logger.warning("Cannot aggregate batches: empty dataframe or missing batch column")
            return pd.DataFrame()
        
        available_params = [p for p in ANALYSIS_PARAMS if p in df.columns]
        
        if not available_params:
            logger.warning("No analysis parameters found in dataframe")
            return df.groupby('batch').size().reset_index(name='Count')
        
        # Define aggregation
        agg_dict = {param: ['mean', 'std', 'median', 'min', 'max', 'count'] for param in available_params}
        
        # Aggregate
        result = df.groupby('batch').agg(agg_dict)
        
        # Flatten column names
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        result = result.reset_index()
        
        logger.info(f"Aggregated {len(result)} batches")
        
        return result
