# AutoIV-Ultimate

**A Comprehensive Python Tool for Automated Solar Cell IV Characteristic Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-xubuxu/AutoIV--Ultimate-blue.svg)](https://github.com/xubuxu/AutoIV-Ultimate)

---

## 🎯 Overview

**AutoIV-Ultimate** is the unified merger of two specialized photovoltaic IV analysis tools, combining the strengths of single-batch deep physics analysis and multi-batch statistical comparison. This professional-grade suite provides researchers and engineers with comprehensive tools for analyzing solar cell Current-Voltage (IV) characteristics, from raw data ingestion to publication-ready reports.

### Merged from Two Specialized Tools:

1. **Auto_IV_Analysis_Suite** - Single-batch analysis with advanced physics modeling
2. **IV_Batch_Analyzer** - Multi-batch statistical comparison with professional GUI

---

## ✨ Key Features

### 🔬 Advanced Physics Modeling (from Auto_IV_Analysis_Suite)

- **Single Diode Model (SDM) Fitting**
  - Lambert W function-based approach for high accuracy
  - Automatic Rs/Rsh extraction
  - Ideality factor (n) calculation
  - R² goodness-of-fit evaluation

- **Two Diode Model (TDM) Analysis**
  - Separate recombination channel identification (J01, J02)
  - Bulk vs interface recombination breakdown
  - Differential evolution optimization

- **S-Shape Detection**
  - Automatic anomaly identification in IV curves
  - Extraction barrier detection (low voltage kink)
  - Injection barrier detection (high voltage rollover)
  - Critical for perovskite and organic solar cells

- **Hysteresis Analysis**
  - Forward/Reverse scan comparison
  - Hysteresis Index (HI) calculation: `HI = (PCE_RS - PCE_FS) / PCE_RS × 100%`
  - Essential for perovskite solar cell characterization

- **Fill Factor Loss Decomposition**
  - FF_ideal calculation
  - Rs-induced losses
  - Rsh-induced losses
  - Recombination losses

### 📊 Multi-Batch Statistical Analysis (from IV_Batch_Analyzer)

- **Automated Data Cleaning**
  - Smart CSV parsing with auto-delimiter detection (comma, semicolon, tab)
  - Automatic header row detection (skips lab equipment metadata)
  - Flexible column mapping (case-insensitive)
  - Duplicate removal with configurable thresholds
  - IQR-based outlier detection and removal

- **Statistical Comparison**
  - Group-wise descriptive statistics (Mean, Median, Std, CV%)
  - Independent t-tests for batch comparison (two-tailed)
  - p-value calculation and significance testing
  - Champion cell identification per batch
  - Top-10 performers ranking

- **Yield Analysis**
  - Total vs valid cell count per batch
  - Efficiency bin distribution (0-5%, 5-10%, 10-15%, 15-20%, >20%)
  - Failure mode classification:
    - Low Efficiency (Eff < threshold)
    - Low Voc (Voc < threshold)
    - Low Jsc (Jsc < threshold)
    - Poor FF (FF < min or > max)

### 🎨 Comprehensive Visualization Suite

#### From Auto_IV_Analysis_Suite (18 plot types):
1. **Yield Statistics** - Total vs valid cells per group
2. **Failure Mode Analysis** - Breakdown of failure reasons
3. **Parameter Distributions** - 6-panel (Jsc, Voc, FF, Eff, Rs, Rsh)
4. **MPPT Stability Analysis** - Maximum power point tracking
5. **Dark IV Analysis** - Leakage current characterization
6. **Multi-Correlation Matrix** - Parameter correlations
7. **S-Shape Detection** - IV curve anomaly visualization
8. **Champion Semi-log IV** - Log-scale current plot
9. **Voc-Jsc Contours** - Iso-efficiency contour maps
10. **Differential Resistance** - dV/dI vs Voltage
11. **Spatial Heatmap** - Efficiency map (if spatial data available)
12. **Champion IV Enhanced** - IV + P-V dual-axis plot
13. **FF Loss Analysis** - Waterfall chart of FF components
14. **Hysteresis Distribution** - HI histogram
15. **Ideality Factor Distribution** - n-factor histogram
16. **Physics Deep Dive** - TDM fitting visualization
17. **Model Fitting** - SDM curve fitting overlay
18. **Anomaly Detection** - Z-score based outlier highlighting

#### From IV_Batch_Analyzer (11 plot types):
1. **Boxplot** - Distribution of Eff, Voc, Jsc, FF, Rs, Rsh across batches
2. **Histogram** - Efficiency distribution with KDE overlay
3. **Trend Analysis** - Batch mean ± std with error bars
4. **Yield Distribution** - Stacked bar chart for efficiency bins
5. **J-V Curves** - Champion cell IV curves per batch
6. **Voc-Jsc Correlation** - Scatter plot with linear fit
7. **Efficiency Drivers** - Multi-panel correlation (Voc, Jsc, FF vs Eff)
8. **Resistance Analysis** - Rs/Rsh box plots (log scale for Rsh)
9. **Model Fitting** - Single-diode model curve overlay
10. **Hysteresis Analysis** - Forward vs Reverse scan comparison
11. **Anomaly Detection** - Statistical outlier highlighting

### 📄 Multi-Format Report Generation

- **Excel Workbook (.xlsx)**
  - Cleaned Data sheet
  - Statistics Summary with CV%
  - Champion Cells per group
  - Top 10 Cells overall
  - Yield Distribution
  - Hysteresis Metrics (if available)
  - Auto-formatted tables with conditional formatting

- **Word Document (.docx)**
  - Executive Summary with t-test results
  - Performance Parameters Table with Group Averages
  - Physics & Hysteresis Summary Table
  - Failure Mode Analysis Table
  - Parameter Correlation Matrix Table
  - Champion Cell Details Table
  - All plots embedded with captions (300 DPI)
  - Professional styling (Calibri, consistent headers)

- **PowerPoint Presentation (.pptx)**
  - Title slide
  - Summary statistics slide
  - Data tables (paginated for large datasets)
  - Full-resolution plot slides (one plot per slide)
  - Professional layout (16:9 aspect ratio)

### 🖥️ Dual UI System

#### 1. Streamlit Web Interface (from Auto_IV_Analysis_Suite)
- Modern, responsive web-based UI
- Real-time parameter adjustment
- Interactive plot previews
- Session state management
- Configuration file persistence (`iv_analysis_config.json`)
- Launch: `streamlit run auto_iv_suite/web_ui.py`

#### 2. CustomTkinter Desktop GUI (from IV_Batch_Analyzer)
- Native desktop application
- Dark/Light theme toggle
- 3-Tab Interface:
  - **Dashboard**: Parameter configuration and analysis control
  - **Live Log**: Real-time color-coded logging (INFO/WARNING/ERROR)
  - **Plot Preview**: Embedded matplotlib viewer with zoom/pan
- Thread-safe execution with stop/cancel functionality
- Auto-saves settings to `~/.iv_analyzer/config.json`
- Launch: `python main.py`

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xubuxu/AutoIV-Ultimate.git
cd AutoIV-Ultimate
```

2. Create virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Streamlit Web UI (Single-Batch Analysis)
```bash
streamlit run auto_iv_suite/web_ui.py
```

### Option 2: CustomTkinter Desktop GUI (Multi-Batch Analysis)
```bash
python iv_batch_main.py
```

### Option 3: Command-Line Interface
```bash
python -m auto_iv_suite --data-folder /path/to/data --config config.json
```

### Option 4: Programmatic Use
```python
from auto_iv_suite import DataLoader, PhysicsEngine, Visualizer, Reporter
from auto_iv_suite.config import Config

# Load configuration
config = Config()

# Load data
loader = DataLoader(config)
data = loader.load_dataset('/path/to/data')

# Perform physics analysis
physics = PhysicsEngine(config)
# ... (analysis code)

# Generate reports
reporter = Reporter(config, output_dir='./output')
reporter.export_all(data, analysis_results)
```

---

## 📁 Project Structure

```
AutoIV-Ultimate/
├── Auto_IV_Analysis_Suite/          # Single-batch analysis module
│   ├── auto_iv_suite/
│   │   ├── __init__.py
│   │   ├── __main__.py              # CLI entry point
│   │   ├── config.py                # Configuration management
│   │   ├── dataloader.py            # Data ingestion (612 lines)
│   │   ├── physics.py               # Physics engines (440 lines)
│   │   │   ├── fit_sdm_model()      # Single Diode Model
│   │   │   ├── fit_tdm_model()      # Two Diode Model
│   │   │   ├── detect_s_shape_type()
│   │   │   └── calculate_ff_losses()
│   │   ├── processor.py             # Core analysis workflow
│   │   ├── visualizer.py            # 18 plot types (914 lines)
│   │   ├── reporter.py              # Excel/Word/PPT generation (28KB)
│   │   ├── web_ui.py                # Streamlit interface
│   │   ├── web_utils.py             # UI utilities
│   │   └── utils.py                 # Logging, helpers
│   ├── requirements.txt
│   └── README.md
│
├── IV_Batch_Analyzer/               # Multi-batch comparison module
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration & constants (9.6KB)
│   │   │   ├── COLUMN_MAPPING       # Flexible CSV column names
│   │   │   ├── ANALYSIS_PARAMS      # Default thresholds
│   │   │   └── PLOT_CONFIG          # Visualization settings
│   │   ├── data_loader.py           # Smart CSV parser (16KB)
│   │   │   ├── detect_delimiter()
│   │   │   ├── detect_header_row()
│   │   │   └── load_all_batches()
│   │   ├── statistics.py            # Statistical analysis (12.6KB)
│   │   │   ├── clean_data()
│   │   │   ├── compute_statistics()
│   │   │   └── calculate_hysteresis_metrics()
│   │   ├── visualizer.py            # 11 plot types (30KB)
│   │   │   ├── Theme support (dark/light)
│   │   │   └── _reconstruct_jv_curve()
│   │   ├── reporter.py              # Report exporters (21KB)
│   │   ├── physics.py               # Single-diode model (16KB)
│   │   └── analyzer.py              # Main controller (7.6KB)
│   ├── main.py                      # CustomTkinter GUI app
│   ├── requirements.txt
│   └── README.md
│
├── requirements.txt                 # Combined dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 📋 Input File Requirements

### Supported Formats
- CSV files (UTF-8, GBK, Latin1 encoding)
- Auto-detected delimiters: comma (`,`), semicolon (`;`), tab (`\t`)
- Automatic header row detection (skips metadata rows)

### Required Columns (case-insensitive)
| Parameter | Accepted Column Names |
|-----------|----------------------|
| Efficiency | `Eff`, `Efficiency`, `PCE`, `Eta` |
| Open Circuit Voltage | `Voc`, `Uoc` |
| Short Circuit Current | `Jsc`, `Isc` |
| Fill Factor | `FF` |

### Optional Columns
| Parameter | Accepted Column Names |
|-----------|----------------------|
| Series Resistance | `Rs` |
| Shunt Resistance | `Rsh`, `Rp` |
| Scan Direction | `ScanDirection`, `Direction` |
| Cell Name | `CellName`, `Device_ID`, `Sample_Name`, `Name` |
| Area | `Area`, `Active_Area` |

### Example CSV Format
```csv
CellName,Voc,Jsc,FF,Eff,Rs,Rsh,ScanDirection
Sample_A1,1.12,24.5,78.2,21.4,2.3,4500,Reverse
Sample_A2,1.10,24.8,76.5,20.8,2.8,3200,Reverse
```

---

## 📊 Output Reports

Analysis generates a timestamped folder (e.g., `IV_Analysis_20250124_143000/`) containing:

### 📊 Excel Workbook (`IV_Processed_Data.xlsx`)
- **Cleaned Data**: Filtered dataset with outliers removed
- **Statistics Summary**: Mean, Median, Std, CV% per group
- **Champion Cells**: Best cell per group
- **Top 10 Cells**: Overall best performers
- **Yield Distribution**: Efficiency bin counts
- **Hysteresis Metrics** (if available): HI, Forward/Reverse PCE

### 📄 Word Report (`IV_Analysis_Report.docx`)
- Executive Summary with t-test p-values
- Performance Parameters Table (with group averages)
- Physics & Hysteresis Summary Table
- Failure Mode Analysis Table
- Parameter Correlation Matrix Table
- Champion Cell Details Table
- All plots embedded at 300 DPI with descriptive captions

### 📽️ PowerPoint Slides (`IV_Analysis_Slides.pptx`)
- Title slide with analysis metadata
- Summary statistics slide
- Data tables (auto-paginated for large datasets)
- Individual plot slides (full-screen, high resolution)

### 🖼️ Plot Files (PNG, 300 DPI)
All plots saved individually for use in publications/presentations.

---

## 🛠️ Configuration

### Auto_IV_Analysis_Suite Configuration (`iv_analysis_config.json`)
```json
{
  "filenames": {
    "raw_data_pattern": "*.txt",
    "summary_pattern": "*.csv"
  },
  "physics": {
    "enable_sdm": true,
    "enable_tdm": true,
    "enable_hysteresis": true,
    "enable_s_shape": true,
    "temperature": 298.15
  },
  "thresholds": {
    "eff_min": 0.5,
    "voc_min": 0.2,
    "jsc_min": 0.5,
    "ff_min": 20.0,
    "ff_max": 90.0,
    "outlier_multiplier": 1.5
  },
  "plotting": {
    "dpi": 300,
    "figure_width": 12,
    "figure_height": 8,
    "font_size": 14,
    "color_palette": "Set2"
  }
}
```

### IV_Batch_Analyzer Configuration (`~/.iv_analyzer/config.json`)
```json
{
  "last_folder": "/path/to/data",
  "scan_direction": "Reverse",
  "thresholds": {
    "Eff_Min": 0.1,
    "Voc_Min": 0.1,
    "Jsc_Min": 0.1,
    "FF_Min": 10.0,
    "FF_Max": 90.0
  },
  "remove_duplicates": true,
  "outlier_removal": true,
  "theme": "dark",
  "window_geometry": "900x700"
}
```

---

## 🔧 Advanced Features

### Physics Modeling Details

#### Single Diode Model Equation
```
J = J₀[exp((V - JRₛ)/(nkT/q)) - 1] + (V - JRₛ)/Rₛₕ - Jₗ
```
Where:
- `J₀`: Saturation current density
- `n`: Ideality factor (1 = ideal, >1 = recombination)
- `Rₛ`: Series resistance (Ω·cm²)
- `Rₛₕ`: Shunt resistance (Ω·cm²)
- `Jₗ`: Photocurrent density

#### Two Diode Model
Separates recombination into:
- `J₀₁`: Bulk recombination (n₁ = 1)
- `J₀₂`: Interface recombination (n₂ = 2)

#### Hysteresis Index (HI)
```
HI = 100% × (PCE_Reverse - PCE_Forward) / PCE_Reverse
```
- HI > 5%: Significant hysteresis (common in perovskites)
- HI < 2%: Negligible hysteresis

### S-Shape Classification
- **Extraction Barrier**: Kink at low voltage (V < Vₘₚₚ)
- **Injection Barrier**: Rollover at high voltage (V > Vₘₚₚ)
- Detected via second derivative analysis of IV curve

---

## 📚 Dependencies

### Core Scientific Stack
- `pandas >= 1.5.0` - Data manipulation
- `numpy >= 1.23.0` - Numerical computations
- `scipy >= 1.9.0` - Scientific algorithms (optimization, stats)

### Visualization
- `matplotlib >= 3.6.0` - Plotting backend
- `seaborn >= 0.12.0` - Statistical visualizations

### Report Generation
- `openpyxl >= 3.0.0` - Excel export
- `python-docx >= 0.8.11` - Word documents
- `python-pptx >= 0.6.21` - PowerPoint slides

### User Interfaces
- `streamlit >= 1.29.0` - Web UI
- `customtkinter >= 5.0.0` - Desktop GUI
- `darkdetect >= 0.8.0` - System theme detection

### Development Tools
- `rich >= 13.0.0` - Enhanced terminal output
- `black >= 23.0.0` - Code formatter
- `isort >= 5.0.0` - Import organizer
- `mypy >= 1.0.0` - Type checker
- `flake8 >= 6.0.0` - Linter

---

## 🔍 Feature Comparison Matrix

| Feature | Auto_IV_Analysis_Suite | IV_Batch_Analyzer | AutoIV-Ultimate |
|---------|----------------------|-------------------|-----------------|
| **UI Type** | Streamlit (Web) | CustomTkinter (Desktop) | Both |
| **Physics Models** | SDM, TDM | SDM | SDM, TDM |
| **S-Shape Detection** | ✅ Yes | ❌ No | ✅ Yes |
| **Hysteresis Analysis** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Multi-Batch t-tests** | ❌ No | ✅ Yes | ✅ Yes |
| **Outlier Removal** | ✅ IQR-based | ✅ IQR-based | ✅ IQR-based |
| **Plot Count** | 18 types | 11 types | 29 types (merged) |
| **Excel Export** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Word Reports** | ✅ Yes | ✅ Yes | ✅ Yes |
| **PowerPoint Reports** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Dark/Light Theme** | ❌ No | ✅ Yes | ✅ Yes |
| **Real-time Logging** | ❌ No | ✅ Yes | ✅ Yes |
| **Configuration Persistence** | ✅ JSON | ✅ JSON | ✅ JSON |
| **CLI Support** | ✅ Yes | ❌ No | ✅ Yes |

---

## 🐛 Troubleshooting

### CSV Parsing Issues
**Problem**: "No valid data found" error

**Solutions**:
- Ensure CSV contains required columns (Eff, Voc, Jsc, FF)
- Check file encoding (try UTF-8, GBK, or Latin1)
- Verify delimiter (auto-detection supports `,`, `;`, `\t`)
- Remove extra metadata rows (tool auto-skips, but check manually if issues persist)

### GUI Not Opening
**Problem**: CustomTkinter window doesn't appear

**Solutions**:
```bash
# Test dependencies
python -c "import customtkinter; print('OK')"

# Check Python version
python --version  # Must be 3.8+

# Reinstall CustomTkinter
pip install --upgrade customtkinter
```

### Physics Fitting Fails
**Problem**: SDM/TDM fitting returns NaN or poor R²

**Solutions**:
- Ensure IV data quality (no missing points, reasonable values)
- Check Voc/Jsc are within expected ranges
- Increase `outlier_multiplier` threshold to remove poor curves
- Verify raw IV data files exist (for detailed fitting)

### Report Generation Errors
**Problem**: Word/PowerPoint export fails

**Solutions**:
```bash
# Reinstall report dependencies
pip install --upgrade python-docx python-pptx openpyxl

# Check write permissions
# Ensure output directory is writable
```

---

## 📖 Documentation References

### Original Project READMEs
- [Auto_IV_Analysis_Suite/README.md](Auto_IV_Analysis_Suite/README.md)
- [IV_Batch_Analyzer/README.md](IV_Batch_Analyzer/README.md)

### Scientific Background
- **Single Diode Model**: Sze & Ng, "Physics of Semiconductor Devices" (Wiley, 2006)
- **Hysteresis in Perovskites**: Snaith et al., J. Phys. Chem. Lett. 2014, 5, 1511
- **S-Shape Curves**: Wagenpfahl et al., Phys. Rev. B 2010, 82, 115306

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern desktop UI
- [Streamlit](https://streamlit.io/) - Web application framework
- [Pandas](https://pandas.pydata.org/) - Data analysis library
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization
- [SciPy](https://scipy.org/) - Scientific computing
- [python-docx](https://python-docx.readthedocs.io/) - Word generation
- [python-pptx](https://python-pptx.readthedocs.io/) - PowerPoint generation

---

## 📞 Support

For issues, questions, or feature requests:
- **GitHub Issues**: [https://github.com/xubuxu/AutoIV-Ultimate/issues](https://github.com/xubuxu/AutoIV-Ultimate/issues)
- **Email**: [Your contact email]

---

## 🚀 Roadmap

### Planned Features
- [ ] Jupyter Notebook integration
- [ ] RESTful API for batch processing
- [ ] Database backend for historical tracking
- [ ] Machine learning-based anomaly detection
- [ ] Interactive Plotly dashboards
- [ ] Multi-language support (Chinese, English)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, Azure, GCP)

---

**AutoIV-Ultimate** - *Making solar cell IV analysis comprehensive, accurate, and beautiful* ☀️

---

## 📸 Screenshots

### Streamlit Web UI
![Streamlit Interface](docs/screenshots/streamlit_ui.png)
*Single-batch analysis with real-time parameter tuning*

### CustomTkinter Desktop GUI
![CustomTkinter GUI](docs/screenshots/customtkinter_gui.png)
*Multi-batch comparison with live logging and plot preview*

### Sample Plots
![Champion IV Curves](docs/screenshots/champion_iv.png)
*Champion cell IV curves with power-voltage overlay*

![Statistical Comparison](docs/screenshots/boxplot.png)
*Statistical parameter distributions across batches*

![Physics Analysis](docs/screenshots/physics_deep_dive.png)
*Two-diode model fitting and recombination analysis*

---

**Last Updated**: 2025-01-24
**Version**: 1.0.0 (Merged Release)
**GitHub**: https://github.com/xubuxu/AutoIV-Ultimate
