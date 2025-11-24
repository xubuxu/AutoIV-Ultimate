# Refined Development Plan for **AutoIV‑Ultimate**

## 🎯 Project Vision
- **Deep Dive** – Precise physical modeling for a single batch (SDM/TDM fitting, S‑Shape detection, hysteresis, failure‑mode analysis).
- **Batch Comparison** – Statistical comparison across multiple batches (box‑plots, histograms, trend lines, T‑tests, yield analysis).
- **Unified GUI** – Modern CustomTkinter interface with dark/light mode, tab‑based workflow, live log, and plot preview.

## 📦 Core Feature Checklist (must‑keep)
| Module | Original Source | Required Capability |
|--------|----------------|----------------------|
| **Data Ingestion** | `Auto_IV_Analysis_Suite/dataloader.py` & `IV_Batch_Analyzer/src/data_loader.py` | Smart delimiter detection, header row skipping, multi‑encoding, raw IV arrays + extracted parameters, threshold filtering, duplicate removal, IQR outlier removal, natural sorting |
| **Physics Engine** | `Auto_IV_Analysis_Suite/physics.py` | SDM (Lambert‑W), TDM (dual‑diode), S‑Shape detection, hysteresis index, failure‑mode analysis |
| **Statistics Engine** | `IV_Batch_Analyzer/src/statistics.py` | Multi‑batch aggregation, group comparison, independent‑sample T‑test, descriptive stats, yield calculation, champion‑cell identification |
| **Visualization Suite** | Both projects | PNG (300 DPI) export of: IV curves, fitted IV, box‑plots, histograms (KDE), trend plots, yield chart, correlation matrix, resistance distribution, hysteresis comparison |
| **Reporting System** | Both projects | Excel (clean data, stats, champion cells, yield), Word (executive summary, tables, images, physics parameters), PowerPoint (title, summary, paginated tables, high‑res plots) |
| **UI/UX** | `IV_Batch_Analyzer/src/ui/` | CustomTkinter dark‑mode UI with three tabs (Dashboard, Live Log, Plot Preview), persistent `config.json`, stop/cancel analysis |

## 🏗️ Proposed Architecture (merged)
```
AutoIV-Ultimate/
├─ main.py                 # CLI entry (launch GUI or Streamlit)
├─ run_gui.py              # Starts CustomTkinter UI
├─ run_streamlit.py        # Starts Streamlit UI (optional)
├─ requirements.txt
├─ README.md
├─ assets/                 # icons, logo
└─ src/
   ├─ __init__.py
   ├─ core/
   │   ├─ config_manager.py   # unified Config class
   │   ├─ data_loader.py      # smart parser + cleaning
   │   └─ logger.py           # thread‑safe logger
   ├─ engines/
   │   ├─ physics.py          # SDM/TDM, S‑Shape, hysteresis
   │   └─ statistics.py       # batch aggregation, T‑test, yield
   ├─ visualization/
   │   ├─ plot_engine.py      # façade exposing plot_* functions
   │   ├─ plot_physics.py     # physics‑specific plots
   │   └─ plot_stats.py       # statistical plots
   ├─ reporting/
   │   ├─ excel_builder.py
   │   ├─ word_builder.py
   │   └─ ppt_builder.py
   └─ ui/
       ├─ app_window.py       # CustomTkinter main window
       ├─ tab_dashboard.py    # config, folder picker, mode switch
       ├─ tab_logs.py         # live log view
       └─ tab_preview.py      # embedded Matplotlib canvas
```

### Key Integration Decisions (need user confirmation)
1. **Dual UI strategy** – Keep both CustomTkinter and Streamlit as separate entry points (`run_gui.py` & `run_streamlit.py`).
2. **Package name** – Use `autoiv_ultimate` for the Python package (import path `autoiv_ultimate.*`).
3. **License** – Both sources are MIT; retain MIT for the merged repo.
4. **Configuration merging** – Consolidate settings from both `config.py` files into a single `Config` dataclass with sections `physics`, `statistics`, `ui`.

## 🛠️ Implementation Steps
| Phase | Tasks | Expected Tool Calls |
|------|-------|---------------------|
| **0 – Prep** | • Initialize Git repo, add `.gitignore`.<br>• Create top‑level `src/` skeleton. | `run_command` (git init) |
| **1 – Core Layer** | • Implement `config_manager.py` (merge configs).<br>• Build unified `data_loader.py` (smart parsing, cleaning).<br>• Add `logger.py`. | file edits (`replace_file_content` / `multi_replace_file_content`) |
| **2 – Engines** | • Port `physics.py` (SDM/TDM, S‑Shape, hysteresis).<br>• Port `statistics.py` (batch aggregation, T‑test, yield). | file creation (`write_to_file`) |
| **3 – Visualization** | • Create `plot_engine.py` façade.<br>• Move physics‑specific plots to `plot_physics.py`.<br>• Move statistical plots to `plot_stats.py`.<br>• Ensure all plots call a common style helper. | file edits & new files |
| **4 – Reporting** | • Consolidate Excel, Word, PPT builders into `reporting/`.<br>• Ensure they accept the unified data objects. | file edits |
| **5 – UI** | • Refactor existing CustomTkinter UI into `ui/` modules.<br>• Add a thin Streamlit wrapper (`run_streamlit.py`).<br>• Wire UI to core engines via the new package imports. | file edits, new files |
| **6 – CLI & Entry Points** | • Write `main.py` that parses `--ui streamlit|gui` and launches the appropriate entry point.<br>• Update `setup.py`/`pyproject.toml` to expose console script. | file edits |
| **7 – Tests & Verification** | • Copy existing unit tests (if any) into `tests/` and adapt imports.<br>• Add integration test: load a sample CSV, run physics engine, generate a minimal report.<br>• Manual sanity‑check of both UIs and generated files. | `run_command` (pytest) |
| **8 – Documentation** | • Overwrite root `README.md` with the vision, feature table, install/run instructions, screenshots (generated via `generate_image`).<br>• Add `CHANGELOG.md`. | `write_to_file` + `generate_image` |
| **9 – GitHub Push** | • Add remote `https://github.com/xubuxu/AutoIV-Ultimate`.<br>• Commit all files, push. | `run_command` (git add/commit/push) |

## ✅ Verification Plan
1. **Automated** – Run the full test suite; ensure 100 % pass.
2. **Manual** –
   - Launch `run_gui.py`; process a single‑batch CSV → verify physics plots & reports.
   - Launch `run_gui.py` in batch mode; verify statistical plots & reports.
   - (Optional) Launch `run_streamlit.py` and repeat the above.
   - Open generated Excel/Word/PPT files; confirm presence of all required sheets/tables/figures.
3. **Git** – Verify `git status` is clean, `git log` shows initial commit, and `git push` succeeds.

---
**Next Steps** (awaiting your confirmation):
- Choose UI strategy (both UIs or primary only).
- Approve package name `autoiv_ultimate`.
- Confirm license remains MIT.

Once approved, I will start implementing Phase 0.
