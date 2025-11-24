"""
AutoIV-Ultimate Physics Engine Module

Ported from Auto_IV_Analysis_Suite/physics.py.
Handles advanced physics modeling: SDM/TDM fitting, S-Shape detection, FF loss analysis.
"""
import logging
import numpy as np
from scipy.special import lambertw
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import linregress
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


# ================= PHYSICS ENGINE =================

class PhysicsEngine:
    """Handles advanced physics modeling (SDM, TDM, S-Shape detection)."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PhysicsEngine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.kT_q = 0.0259  # Thermal voltage at 300K (kT/q in Volts)
        self.logger = logger
        
        logger.info("PhysicsEngine initialized")
    
    # ================= SINGLE DIODE MODEL (SDM) =================
    
    def _sdm_lambertw_func(self, v: Union[float, np.ndarray], jl: float, j0: float, 
                           n: float, rs: float, rsh: float) -> Union[float, np.ndarray]:
        """
        Single Diode Model function using Lambert W function.
        
        Equation:
            J = (Rsh*(Jl + J0) - V) / (Rs + Rsh) - (n*kT/q / Rs) * W(term)
        
        Args:
            v: Voltage array or scalar
            jl: Photocurrent
            j0: Saturation current
            n: Ideality factor
            rs: Series resistance
            rsh: Shunt resistance
            
        Returns:
            Calculated current density
        """
        # Prevent numerical issues
        if rs < 1e-4:
            rs = 1e-4
        if rsh < 1e-2:
            rsh = 1e-2
        if n < 0.1:
            n = 0.1
        
        term1 = (rsh * (jl + j0) - v) / (rs + rsh)
        term2 = (rsh * j0) / (n * self.kT_q * (1 + rs / rsh))
        arg_exp = (rsh * (v + rs * (jl + j0))) / (n * self.kT_q * (rs + rsh))
        
        # Clip to prevent overflow
        arg_exp = np.clip(arg_exp, -50, 100)
        
        w_arg = term2 * np.exp(arg_exp)
        w_val = np.real(lambertw(w_arg))
        
        return term1 - (n * self.kT_q / rs) * w_val
    
    def fit_sdm_model(self, v: np.ndarray, j: np.ndarray, voc: float, jsc: float) -> Tuple[float, float, float, float, float]:
        """
        Fit Single Diode Model to IV data using Lambert W function.
        
        Args:
            v: Voltage array
            j: Current density array
            voc: Open circuit voltage
            jsc: Short circuit current density
            
        Returns:
            Tuple of (j0, n, rs, rsh, r2)
                j0: Saturation current
                n: Ideality factor
                rs: Series resistance
                rsh: Shunt resistance
                r2: R-squared goodness of fit
        """
        try:
            # Estimate Rs from slope near Voc
            mask_voc = (v > voc - 0.1) & (v < voc + 0.1)
            if np.sum(mask_voc) > 2:
                slope_voc = linregress(v[mask_voc], j[mask_voc])[0]
                rs_guess = abs(1 / slope_voc) if slope_voc != 0 else 2.0
            else:
                rs_guess = 2.0
            
            # Initial guess: [jl, j0, n, rs, rsh]
            p0 = [jsc, 1e-9, 1.5, np.clip(rs_guess, 0.01, 50), 1000.0]
            
            # Bounds
            bounds = (
                [jsc * 0.8, 1e-15, 0.5, 1e-4, 1.0],
                [jsc * 1.2, 1e-3, 5.0, 100.0, 1e7]
            )
            
            # Fit using curve_fit
            popt, _ = curve_fit(
                self._sdm_lambertw_func, v, j,
                p0=p0, bounds=bounds, maxfev=3000
            )
            
            # Calculate R²
            j_pred = self._sdm_lambertw_func(v, *popt)
            ss_res = np.sum((j - j_pred) ** 2)
            ss_tot = np.sum((j - np.mean(j)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 < 0.9:
                raise ValueError("Poor fit (R² < 0.9)")
            
            # Return j0, n, rs, rsh, r2
            return popt[1], popt[2], popt[3], popt[4], r2
            
        except Exception as e:
            # Fallback: Use differential evolution for global optimization
            logger.debug(f"curve_fit failed, trying differential_evolution: {e}")
            
            try:
                def cost_func(params):
                    jl, j0, n, rs, rsh = params
                    j_model = self._sdm_lambertw_func(v, jl, j0, n, rs, rsh)
                    return np.sum((j - j_model) ** 2)
                
                de_bounds = list(zip(bounds[0], bounds[1]))
                result = differential_evolution(
                    cost_func, de_bounds,
                    strategy='best1bin',
                    maxiter=100,
                    popsize=10,
                    tol=0.01
                )
                
                if result.success:
                    # Refine with curve_fit
                    popt_refined, _ = curve_fit(
                        self._sdm_lambertw_func, v, j,
                        p0=result.x, bounds=bounds, maxfev=2000
                    )
                    
                    j_pred = self._sdm_lambertw_func(v, *popt_refined)
                    r2 = 1 - (np.sum((j - j_pred) ** 2) / np.sum((j - np.mean(j)) ** 2))
                    
                    return popt_refined[1], popt_refined[2], popt_refined[3], popt_refined[4], r2
                else:
                    return np.nan, np.nan, np.nan, np.nan, 0.0
                    
            except Exception as e2:
                logger.debug(f"SDM fit failed completely: {e2}")
                return np.nan, np.nan, np.nan, np.nan, 0.0
    
    # ================= TWO DIODE MODEL (TDM) =================
    
    def _tdm_residual(self, params: list, v: np.ndarray, j_exp: np.ndarray) -> np.ndarray:
        """
        Residual function for Two Diode Model.
        
        TDM Equation:
            J = Jph - J01*(exp(q*V/(n1*kT)) - 1) - J02*(exp(q*V/(n2*kT)) - 1) - V/Rsh
        """
        jph, j01, j02, rs, rsh = params
        
        # Simplified TDM (ignoring Rs for residual calculation)
        j_model = jph - j01 * (np.exp(v / (1.0 * self.kT_q)) - 1) - j02 * (np.exp(v / (2.0 * self.kT_q)) - 1) - v / rsh
        
        return j_exp - j_model
    
    def fit_tdm_model(self, v: np.ndarray, j: np.ndarray, voc: float, jsc: float, 
                      rs_guess: float = 1.0, rsh_guess: float = 1000.0) -> Tuple[float, float, float, float, float, float]:
        """
        Fit Two Diode Model to IV data.
        
        Args:
            v: Voltage array
            j: Current density array
            voc: Open circuit voltage
            jsc: Short circuit current density
            rs_guess: Initial guess for series resistance
            rsh_guess: Initial guess for shunt resistance
            
        Returns:
            Tuple of (jph, j01, j02, rs, rsh, r2)
                jph: Photocurrent
                j01: Saturation current (diode 1)
                j02: Saturation current (diode 2)
                rs: Series resistance
                rsh: Shunt resistance
                r2: R-squared goodness of fit
        """
        try:
            # Initial guess: [jph, j01, j02, rs, rsh]
            p0 = [jsc, 1e-10, 1e-8, rs_guess, rsh_guess]
            
            # Bounds
            bounds = (
                [jsc * 0.8, 1e-15, 1e-15, 1e-4, 1.0],
                [jsc * 1.2, 1e-5, 1e-3, 100.0, 1e7]
            )
            
            def cost_func(params):
                residuals = self._tdm_residual(params, v, j)
                return np.sum(residuals ** 2)
            
            de_bounds = list(zip(bounds[0], bounds[1]))
            result = differential_evolution(
                cost_func, de_bounds,
                strategy='best1bin',
                maxiter=150,
                popsize=15,
                tol=0.01
            )
            
            if result.success:
                popt = result.x
                j_pred = j - self._tdm_residual(popt, v, j)
                r2 = 1 - (np.sum((j - j_pred) ** 2) / np.sum((j - np.mean(j)) ** 2))
                
                return popt[0], popt[1], popt[2], popt[3], popt[4], r2
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, 0.0
                
        except Exception as e:
            logger.debug(f"TDM fit failed: {e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, 0.0
    
    # ================= S-SHAPE DETECTION =================
    
    def detect_s_shape_type(self, v: np.ndarray, j: np.ndarray, voc: float, jsc: float, ff: float) -> str:
        """
        Detect S-shape type in IV curve using curvature analysis.
        
        S-shapes indicate charge extraction/injection barriers in solar cells.
        
        Args:
            v: Voltage array
            j: Current density array
            voc: Open circuit voltage
            jsc: Short circuit current density
            ff: Fill factor
            
        Returns:
            'None', 'Extraction Barrier', or 'Injection Barrier'
        """
        try:
            # Focus on region between 5% and 95% of Voc
            mask = (v > 0.05) & (v < voc * 0.95)
            if np.sum(mask) < 10:
                return "None"
            
            v_roi = v[mask]
            j_roi = j[mask]
            
            # Sort by voltage
            sort_idx = np.argsort(v_roi)
            v_s = v_roi[sort_idx]
            j_s = j_roi[sort_idx]
            
            # Normalize
            v_norm = (v_s - v_s.min()) / (v_s.max() - v_s.min())
            j_norm = (j_s - j_s.min()) / (j_s.max() - j_s.min())
            
            # Smooth using Savitzky-Golay filter
            win_len = max(5, len(v_norm) // 10)
            if win_len % 2 == 0:
                win_len += 1
            
            j_smooth = savgol_filter(j_norm, window_length=win_len, polyorder=3)
            
            # Calculate first and second derivatives
            d1 = np.gradient(j_smooth, v_norm)
            d2 = np.gradient(d1, v_norm)
            
            # Thresholds for S-shape detection
            min_curvature = -0.5  # Inflection point threshold
            peak_curvature = 10.0  # Kink threshold
            
            has_inflection = np.min(d2) < min_curvature
            has_kink = np.max(d2) > peak_curvature
            
            # S-shape typically correlates with low FF
            if (has_inflection or has_kink) and ff < 75.0:
                # Determine type based on location
                idx_peak = np.argmax(np.abs(d2))
                v_loc = v_s[idx_peak]
                
                if v_loc < voc * 0.5:
                    return "Extraction Barrier"
                else:
                    return "Injection Barrier"
                    
        except Exception as e:
            logger.debug(f"S-shape detection failed: {e}")
        
        return "None"
    
    # ================= FILL FACTOR LOSS ANALYSIS =================
    
    def calculate_ff_losses(self, voc: float, jsc: float, ff_measured: float, rs: float, rsh: float) -> Dict[str, float]:
        """
        Calculate FF loss breakdown into components.
        
        FF_measured = FF_ideal - FF_rs_loss - FF_rsh_loss - FF_recomb_loss
        
        Args:
            voc: Open circuit voltage
            jsc: Short circuit current density
            ff_measured: Measured fill factor (%)
            rs: Series resistance
            rsh: Shunt resistance
            
        Returns:
            Dictionary with FF_ideal, FF_rs_loss, FF_rsh_loss, FF_recomb_loss
        """
        try:
            # Normalized Voc
            voc_norm = voc / self.kT_q
            
            # Ideal FF (Green 1982 approximation)
            ff_ideal = (voc_norm - np.log(voc_norm + 0.72)) / (voc_norm + 1)
            ff_ideal_pct = ff_ideal * 100
            
            # Series resistance loss
            rs_norm = rs * jsc / voc
            ff_rs_loss = ff_ideal * rs_norm * (voc_norm + 1)
            ff_rs_loss_pct = ff_rs_loss * 100
            
            # Shunt resistance loss
            rsh_norm = rsh * jsc / voc
            if rsh_norm > 0:
                ff_rsh_loss = ff_ideal * (voc_norm / rsh_norm)
                ff_rsh_loss_pct = ff_rsh_loss * 100
            else:
                ff_rsh_loss_pct = 0.0
            
            # Recombination loss (residual)
            ff_recomb_loss_pct = ff_ideal_pct - ff_measured - ff_rs_loss_pct - ff_rsh_loss_pct
            
            return {
                'FF_ideal': ff_ideal_pct,
                'FF_rs_loss': ff_rs_loss_pct,
                'FF_rsh_loss': ff_rsh_loss_pct,
                'FF_recomb_loss': max(0, ff_recomb_loss_pct)
            }
            
        except Exception as e:
            logger.debug(f"FF loss calculation failed: {e}")
            return {
                'FF_ideal': np.nan,
                'FF_rs_loss': np.nan,
                'FF_rsh_loss': np.nan,
                'FF_recomb_loss': np.nan
            }
    
    # ================= UTILITY METHODS =================
    
    def calculate_ideality_factor(self, v: np.ndarray, j: np.ndarray, voc: float, jsc: float) -> float:
        """
        Calculate ideality factor from dark IV curve or light IV near Voc.
        
        Args:
            v: Voltage array
            j: Current density array
            voc: Open circuit voltage
            jsc: Short circuit current density
            
        Returns:
            Ideality factor (n)
        """
        try:
            # Use region near Voc (0.5*Voc to 0.9*Voc)
            mask = (v > 0.5 * voc) & (v < 0.9 * voc)
            if np.sum(mask) < 5:
                return np.nan
            
            v_fit = v[mask]
            j_fit = j[mask]
            
            # Convert to log scale (j = j0 * exp(qV/nkT))
            # ln(j) = ln(j0) + qV/(nkT)
            j_abs = np.abs(j_fit - jsc)
            j_abs = np.clip(j_abs, 1e-10, None)  # Avoid log(0)
            
            ln_j = np.log(j_abs)
            
            # Linear fit: slope = q/(nkT)
            slope, intercept = np.polyfit(v_fit, ln_j, 1)
            
            # n = q / (slope * kT)
            q = 1.602e-19  # Elementary charge
            k = 1.381e-23  # Boltzmann constant
            T = 300  # Temperature (K)
            
            n = q / (slope * k * T)
            
            return max(1.0, min(5.0, n))  # Clamp to reasonable range
            
        except Exception as e:
            logger.debug(f"Ideality factor calculation failed: {e}")
            return np.nan
