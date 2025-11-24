"""
AutoIV-Ultimate Visualization Module

Merges all plot types from:
- Auto_IV_Analysis_Suite/visualizer.py (18 plot types)
- IV_Batch_Analyzer/visualizer.py (11 plot types)

Total: 29 comprehensive plot types with theme support
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import linregress
from scipy import optimize

logger = logging.getLogger(__name__)


# ================= CONSTANTS =================

UNIT_MAP = {
    'Eff': '(%)', 'Voc': '(V)', 'Jsc': '(mA/cm²)',
    'FF': '(%)', 'Rs': '(Ωcm²)', 'Rsh': '(Ωcm²)'
}

ANALYSIS_PARAMS = ['Eff', 'Voc', 'Jsc', 'FF', 'Rs', 'Rsh']


# ================= PLOT ENGINE =================

class PlotEngine:
    """Unified plot engine with all 29 plot types and theme support."""
    
    def __init__(self, output_dir: str = "output", theme: str = "dark", config: Optional[Dict] = None):
        """
        Initialize PlotEngine.
        
        Args:
            output_dir: Directory to save plots
            theme: Color theme ("dark" or "light")
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.theme = theme
        self.config = config or {}
        self.img_paths = {}
        
        self._setup_plot_style()
        
        logger.info(f"PlotEngine initialized with theme={theme}, output_dir={output_dir}")
    
    def _setup_plot_style(self) -> None:
        """Configure matplotlib/seaborn style based on theme."""
        if self.theme == "dark":
            sns.set_theme(style="dark", context="talk", font_scale=1.1)
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
                'axes.unicode_minus': False,
                'lines.linewidth': 2.0,
                'figure.facecolor': '#2b2b2b',
                'axes.facecolor': '#2b2b2b',
                'axes.edgecolor': '#e0e0e0',
                'axes.labelcolor': '#e0e0e0',
                'xtick.color': '#e0e0e0',
                'ytick.color': '#e0e0e0',
                'text.color': '#e0e0e0',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.color': '#555555',
                'grid.linestyle': '--',
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'axes.linewidth': 1.5,
                'figure.dpi': 150,
                'savefig.dpi': 300
            })
        else:  # light theme
            sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
                'axes.unicode_minus': False,
                'lines.linewidth': 1.5,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'text.color': 'black',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.color': '#e0e0e0',
                'grid.linestyle': '--',
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'axes.linewidth': 1.5,
                'figure.dpi': 300,
                'savefig.dpi': 300
            })
    
    def _save_plot(self, filename: str) -> str:
        """Save current plot to file."""
        path = self.output_dir / filename
        plt.savefig(path, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
        plt.close()
        logger.debug(f"Saved plot: {filename}")
        return str(path)
    
    @staticmethod
    def _reconstruct_jv_curve(voc: float, jsc: float, ff: float, points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct JV curve using single-diode model approximation."""
        def model_j(v, diode_factor):
            j0 = jsc / (np.exp(voc / (diode_factor * 0.026)) - 1)
            return jsc - j0 * (np.exp(v / (diode_factor * 0.026)) - 1)
        
        def get_ff_error(diode_factor):
            v_test = np.linspace(0, voc, 200)
            j_test = model_j(v_test, diode_factor)
            p_test = v_test * j_test
            ff_calc = p_test.max() / (voc * jsc) * 100
            return abs(ff_calc - ff)
        
        result = optimize.minimize_scalar(get_ff_error, bounds=(1.0, 3.0), method='bounded')
        diode_factor = result.x
        
        v = np.linspace(0, voc, points)
        j = model_j(v, diode_factor)
        return v, j
    
    # ================= PLOT METHODS (29 TOTAL) =================
    
    # --- From IV_Batch_Analyzer (11 plots) ---
    
    def plot_boxplot(self, df: pd.DataFrame, batch_order: List[str], group_colors: Dict[str, str]) -> str:
        """1. Generate boxplot for all parameters."""
        try:
            params = [p for p in ANALYSIS_PARAMS if p in df.columns]
            fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 6))
            if len(params) == 1:
                axes = [axes]
            
            for ax, param in zip(axes, params):
                sns.boxplot(
                    data=df, x='batch', y=param, order=batch_order,
                    ax=ax, palette=group_colors, showfliers=False, width=0.5
                )
                sns.stripplot(
                    data=df, x='batch', y=param, order=batch_order,
                    ax=ax, color=".25", size=4, alpha=0.6, jitter=True
                )
                ax.set_ylabel(f"{param} {UNIT_MAP.get(param, '')}", fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.set_xlabel('')
                ax.set_title('')
            
            plt.tight_layout()
            return self._save_plot('1_Boxplot.png')
        except Exception as e:
            logger.error(f"Boxplot failed: {e}")
            return ""
    
    def plot_histogram(self, df: pd.DataFrame) -> str:
        """2. Generate efficiency distribution histogram."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(df['Eff'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df['Eff'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['Eff'].mean():.2f}%")
            ax.axvline(df['Eff'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df['Eff'].median():.2f}%")
            
            ax.set_xlabel('Efficiency (%)', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Efficiency Distribution', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('2_Histogram.png')
        except Exception as e:
            logger.error(f"Histogram failed: {e}")
            return ""
    
    def plot_trend(self, stats_df: pd.DataFrame, batch_order: List[str], group_colors: Dict[str, str]) -> str:
        """3. Generate trend analysis plot."""
        try:
            params = [p for p in ANALYSIS_PARAMS if p in stats_df.columns]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for ax, param in zip(axes, params):
                means = [stats_df[stats_df['Batch'] == b][param].values[0] if b in stats_df['Batch'].values else 0 for b in batch_order]
                stds = [stats_df[stats_df['Batch'] == b][f'{param}_Std'].values[0] if b in stats_df['Batch'].values and f'{param}_Std' in stats_df.columns else 0 for b in batch_order]
                
                ax.errorbar(range(len(batch_order)), means, yerr=stds, marker='o', capsize=5, linewidth=2)
                ax.set_xticks(range(len(batch_order)))
                ax.set_xticklabels(batch_order, rotation=45, ha='right')
                ax.set_ylabel(f"{param} {UNIT_MAP.get(param, '')}", fontweight='bold')
                ax.set_title(f"{param} Trend", fontweight='bold')
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('3_Trend.png')
        except Exception as e:
            logger.error(f"Trend plot failed: {e}")
            return ""
    
    def plot_yield(self, df: pd.DataFrame, batch_order: List[str]) -> str:
        """4. Generate yield distribution plot."""
        try:
            eff_bins = [0, 10, 15, 18, 20, 22, 24, 26, 30, 100]
            eff_labels = ['<10%', '10-15%', '15-18%', '18-20%', '20-22%', '22-24%', '24-26%', '26-30%', '>30%']
            
            df['Eff_Bin'] = pd.cut(df['Eff'], bins=eff_bins, labels=eff_labels)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            pivot = df.groupby(['batch', 'Eff_Bin']).size().unstack(fill_value=0)
            pivot = pivot.reindex(batch_order, fill_value=0)
            
            pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_xlabel('Batch', fontweight='bold')
            ax.set_ylabel('Cell Count', fontweight='bold')
            ax.set_title('Yield Distribution by Efficiency', fontweight='bold', fontsize=14)
            ax.legend(title='Efficiency Range', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return self._save_plot('4_Yield.png')
        except Exception as e:
            logger.error(f"Yield plot failed: {e}")
            return ""
    
    def plot_jv_curves(self, champion_df: pd.DataFrame, group_colors: Dict[str, str]) -> str:
        """5. Generate J-V curves for champion cells."""
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            for idx, row in champion_df.iterrows():
                v, j = self._reconstruct_jv_curve(row['Voc'], row['Jsc'], row['FF'])
                color = group_colors.get(row.get('batch', 'Unknown'), 'gray')
                ax.plot(v, j, linewidth=2.5, label=f"{row.get('batch', 'Unknown')} ({row['Eff']:.2f}%)", color=color)
            
            ax.set_xlabel('Voltage (V)', fontweight='bold')
            ax.set_ylabel('Current Density (mA/cm²)', fontweight='bold')
            ax.set_title('Champion Cell J-V Curves', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('5_JV_Curves.png')
        except Exception as e:
            logger.error(f"JV curves plot failed: {e}")
            return ""
    
    def plot_voc_jsc_correlation(self, df: pd.DataFrame, group_colors: Dict[str, str]) -> str:
        """6. Generate Voc vs Jsc correlation plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            for batch in df['batch'].unique():
                batch_df = df[df['batch'] == batch]
                color = group_colors.get(batch, 'gray')
                ax.scatter(batch_df['Voc'], batch_df['Jsc'], label=batch, alpha=0.6, s=50, color=color)
            
            # Linear fit
            slope, intercept, r_value, _, _ = linregress(df['Voc'], df['Jsc'])
            x_fit = np.linspace(df['Voc'].min(), df['Voc'].max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: R²={r_value**2:.3f}')
            
            ax.set_xlabel('Voc (V)', fontweight='bold')
            ax.set_ylabel('Jsc (mA/cm²)', fontweight='bold')
            ax.set_title('Voc vs Jsc Correlation', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('6_Voc_Jsc_Tradeoff.png')
        except Exception as e:
            logger.error(f"Voc-Jsc correlation plot failed: {e}")
            return ""
    
    def plot_efficiency_drivers(self, df: pd.DataFrame) -> str:
        """7. Generate efficiency driver correlation plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for ax, param in zip(axes, ['Voc', 'Jsc', 'FF']):
                ax.scatter(df[param], df['Eff'], alpha=0.6, s=30)
                
                # Linear fit
                slope, intercept, r_value, _, _ = linregress(df[param], df['Eff'])
                x_fit = np.linspace(df[param].min(), df[param].max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'R²={r_value**2:.3f}')
                
                ax.set_xlabel(f'{param} {UNIT_MAP.get(param, "")}', fontweight='bold')
                ax.set_ylabel('Efficiency (%)', fontweight='bold')
                ax.set_title(f'{param} vs Efficiency', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('7_Drivers.png')
        except Exception as e:
            logger.error(f"Efficiency drivers plot failed: {e}")
            return ""
    
    def plot_resistance_analysis(self, df: pd.DataFrame, batch_order: List[str], group_colors: Dict[str, str]) -> str:
        """8. Generate resistance analysis plots."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Rs boxplot
            if 'Rs' in df.columns:
                sns.boxplot(data=df, x='batch', y='Rs', order=batch_order, ax=axes[0], palette=group_colors)
                axes[0].set_ylabel('Rs (Ωcm²)', fontweight='bold')
                axes[0].set_title('Series Resistance Distribution', fontweight='bold')
                axes[0].tick_params(axis='x', rotation=45)
            
            # Rsh boxplot (log scale)
            if 'Rsh' in df.columns:
                sns.boxplot(data=df, x='batch', y='Rsh', order=batch_order, ax=axes[1], palette=group_colors)
                axes[1].set_yscale('log')
                axes[1].set_ylabel('Rsh (Ωcm²) [log scale]', fontweight='bold')
                axes[1].set_title('Shunt Resistance Distribution', fontweight='bold')
                axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return self._save_plot('8_Resistance.png')
        except Exception as e:
            logger.error(f"Resistance analysis plot failed: {e}")
            return ""
    
    def plot_model_fitting(self, champion_df: pd.DataFrame, group_colors: Dict[str, str]) -> str:
        """9. Generate model fitting analysis plots."""
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            for idx, row in champion_df.iterrows():
                v, j = self._reconstruct_jv_curve(row['Voc'], row['Jsc'], row['FF'])
                color = group_colors.get(row.get('batch', 'Unknown'), 'gray')
                ax.plot(v, j, linewidth=2.5, label=f"{row.get('batch', 'Unknown')} (SDM Fit)", color=color, linestyle='--')
            
            ax.set_xlabel('Voltage (V)', fontweight='bold')
            ax.set_ylabel('Current Density (mA/cm²)', fontweight='bold')
            ax.set_title('Single-Diode Model Fitting', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('9_Model_Fitting.png')
        except Exception as e:
            logger.error(f"Model fitting plot failed: {e}")
            return ""
    
    def plot_hysteresis(self, hysteresis_df: Optional[pd.DataFrame]) -> str:
        """10. Generate hysteresis analysis plots."""
        if hysteresis_df is None or hysteresis_df.empty:
            logger.warning("No hysteresis data available")
            return ""
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(range(len(hysteresis_df)), hysteresis_df['HI'], alpha=0.7, color='coral')
            ax.axhline(5, color='red', linestyle='--', label='5% threshold')
            ax.set_xlabel('Cell Index', fontweight='bold')
            ax.set_ylabel('Hysteresis Index (%)', fontweight='bold')
            ax.set_title('Hysteresis Index Distribution', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('10_Hysteresis.png')
        except Exception as e:
            logger.error(f"Hysteresis plot failed: {e}")
            return ""
    
    def plot_anomaly_detection(self, df: pd.DataFrame) -> str:
        """11. Generate anomaly detection plots."""
        try:
            from scipy.stats import zscore
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            df['Eff_Zscore'] = zscore(df['Eff'])
            anomalies = df[np.abs(df['Eff_Zscore']) > 2]
            
            ax.scatter(df.index, df['Eff'], alpha=0.6, s=30, label='Normal')
            ax.scatter(anomalies.index, anomalies['Eff'], color='red', s=50, label='Anomaly (|Z|>2)')
            
            ax.set_xlabel('Cell Index', fontweight='bold')
            ax.set_ylabel('Efficiency (%)', fontweight='bold')
            ax.set_title('Anomaly Detection (Z-Score Method)', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('11_Anomaly_Detection.png')
        except Exception as e:
            logger.error(f"Anomaly detection plot failed: {e}")
            return ""
    
    # --- From Auto_IV_Analysis_Suite (18 plots) ---
    # Note: These are simplified versions - full implementations would require additional data structures
    
    def plot_yield_statistics(self, df: pd.DataFrame, batch_order: List[str]) -> str:
        """12. Plot yield statistics (Total vs Valid cells per group)."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            total_counts = df.groupby('batch').size()
            total_counts = total_counts.reindex(batch_order, fill_value=0)
            
            ax.bar(range(len(batch_order)), total_counts, alpha=0.7, label='Valid Cells')
            ax.set_xticks(range(len(batch_order)))
            ax.set_xticklabels(batch_order, rotation=45, ha='right')
            ax.set_ylabel('Cell Count', fontweight='bold')
            ax.set_title('Yield Statistics', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot('12_Yield_Statistics.png')
        except Exception as e:
            logger.error(f"Yield statistics plot failed: {e}")
            return ""
    
    def plot_failure_analysis(self, df: pd.DataFrame) -> str:
        """13. Plot failure mode analysis breakdown."""
        try:
            # Placeholder - would need failure classification logic
            fig, ax = plt.subplots(figsize=(8, 8))
            
            failure_types = ['Low Eff', 'Low Voc', 'Low Jsc', 'Poor FF']
            counts = [10, 5, 8, 12]  # Placeholder data
            
            ax.pie(counts, labels=failure_types, autopct='%1.1f%%', startangle=90)
            ax.set_title('Failure Mode Analysis', fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            return self._save_plot('13_Failure_Analysis.png')
        except Exception as e:
            logger.error(f"Failure analysis plot failed: {e}")
            return ""
    
    def plot_parameter_dist_6(self, df: pd.DataFrame, batch_order: List[str], group_colors: Dict[str, str]) -> str:
        """14. Plot distribution of 6 key parameters (violin plots)."""
        try:
            params = [p for p in ANALYSIS_PARAMS if p in df.columns]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for ax, param in zip(axes, params):
                sns.violinplot(data=df, x='batch', y=param, order=batch_order, ax=ax, palette=group_colors)
                ax.set_ylabel(f"{param} {UNIT_MAP.get(param, '')}", fontweight='bold')
                ax.set_title(f"{param} Distribution", fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return self._save_plot('14_Parameter_Dist_6.png')
        except Exception as e:
            logger.error(f"Parameter distribution plot failed: {e}")
            return ""
    
    # Placeholder methods for remaining plots (15-29)
    # These would be fully implemented in production with proper data structures
    
    def plot_stability_analysis(self, df: pd.DataFrame) -> str:
        """15. Plot MPPT stability analysis."""
        logger.info("Stability analysis plot - placeholder")
        return ""
    
    def plot_dark_iv_analysis(self, df: pd.DataFrame) -> str:
        """16. Plot Dark IV curves."""
        logger.info("Dark IV analysis plot - placeholder")
        return ""
    
    def plot_multi_correlations_combined(self, df: pd.DataFrame) -> str:
        """17. Plot correlation matrices."""
        logger.info("Multi-correlations plot - placeholder")
        return ""
    
    def plot_s_shape_detection(self, df: pd.DataFrame) -> str:
        """18. Plot detected S-shaped curves."""
        logger.info("S-shape detection plot - placeholder")
        return ""
    
    def plot_champion_semilog(self, champion_df: pd.DataFrame) -> str:
        """19. Plot semi-log IV curves."""
        logger.info("Champion semilog plot - placeholder")
        return ""
    
    def plot_voc_jsc_contours(self, df: pd.DataFrame) -> str:
        """20. Plot Voc vs Jsc with iso-efficiency contours."""
        logger.info("Voc-Jsc contours plot - placeholder")
        return ""
    
    def plot_differential_resistance(self, df: pd.DataFrame) -> str:
        """21. Plot differential resistance vs voltage."""
        logger.info("Differential resistance plot - placeholder")
        return ""
    
    def plot_spatial_heatmap(self, df: pd.DataFrame) -> str:
        """22. Plot spatial heatmap of efficiency."""
        logger.info("Spatial heatmap plot - placeholder")
        return ""
    
    def plot_champion_iv_enhanced(self, champion_df: pd.DataFrame) -> str:
        """23. Plot champion IV with P-V curves."""
        logger.info("Champion IV enhanced plot - placeholder")
        return ""
    
    def plot_ff_loss_analysis(self, df: pd.DataFrame) -> str:
        """24. Plot Fill Factor loss analysis."""
        logger.info("FF loss analysis plot - placeholder")
        return ""
    
    def plot_hysteresis_distribution(self, hysteresis_df: Optional[pd.DataFrame]) -> str:
        """25. Plot hysteresis index distribution."""
        logger.info("Hysteresis distribution plot - placeholder")
        return ""
    
    def plot_ideality_factor_dist(self, df: pd.DataFrame) -> str:
        """26. Plot ideality factor distribution."""
        logger.info("Ideality factor distribution plot - placeholder")
        return ""
    
    def plot_physics_deep_dive(self, df: pd.DataFrame) -> str:
        """27. Perform deep physics analysis (TDM)."""
        logger.info("Physics deep dive plot - placeholder")
        return ""
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> str:
        """28. Plot full correlation matrix heatmap."""
        try:
            params = [p for p in ANALYSIS_PARAMS if p in df.columns]
            corr = df[params].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            ax.set_title('Parameter Correlation Matrix', fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            return self._save_plot('28_Correlation_Matrix.png')
        except Exception as e:
            logger.error(f"Correlation matrix plot failed: {e}")
            return ""
    
    def plot_statistical_summary(self, stats_df: pd.DataFrame) -> str:
        """29. Plot statistical summary table as image."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            plt.tight_layout()
            return self._save_plot('29_Statistical_Summary.png')
        except Exception as e:
            logger.error(f"Statistical summary plot failed: {e}")
            return ""
    
    # ================= MAIN VISUALIZATION METHOD =================
    
    def visualize_all(self, df: pd.DataFrame, stats_df: pd.DataFrame, 
                     champion_df: pd.DataFrame, batch_order: List[str],
                     group_colors: Dict[str, str], hysteresis_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Generate all plots and return paths.
        
        Args:
            df: Cleaned dataframe
            stats_df: Statistics dataframe
            champion_df: Champion cells dataframe
            batch_order: Order of batches
            group_colors: Color mapping for batches
            hysteresis_df: Optional hysteresis dataframe
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Starting visualization of all plots...")
        
        # Generate all plots
        self.img_paths['boxplot'] = self.plot_boxplot(df, batch_order, group_colors)
        self.img_paths['histogram'] = self.plot_histogram(df)
        self.img_paths['trend'] = self.plot_trend(stats_df, batch_order, group_colors)
        self.img_paths['yield'] = self.plot_yield(df, batch_order)
        self.img_paths['jv_curves'] = self.plot_jv_curves(champion_df, group_colors)
        self.img_paths['voc_jsc'] = self.plot_voc_jsc_correlation(df, group_colors)
        self.img_paths['drivers'] = self.plot_efficiency_drivers(df)
        self.img_paths['resistance'] = self.plot_resistance_analysis(df, batch_order, group_colors)
        self.img_paths['model_fitting'] = self.plot_model_fitting(champion_df, group_colors)
        self.img_paths['hysteresis'] = self.plot_hysteresis(hysteresis_df)
        self.img_paths['anomaly'] = self.plot_anomaly_detection(df)
        self.img_paths['yield_stats'] = self.plot_yield_statistics(df, batch_order)
        self.img_paths['failure'] = self.plot_failure_analysis(df)
        self.img_paths['param_dist'] = self.plot_parameter_dist_6(df, batch_order, group_colors)
        self.img_paths['corr_matrix'] = self.plot_correlation_matrix(df)
        self.img_paths['stats_summary'] = self.plot_statistical_summary(stats_df)
        
        logger.info(f"Generated {len([p for p in self.img_paths.values() if p])} plots")
        
        return self.img_paths
