"""
AutoIV-Ultimate Main Application Window

Complete CustomTkinter UI with Dashboard, Live Log, and Plot Preview tabs.
"""
import customtkinter as ctk
from tkinter import filedialog
import logging
from pathlib import Path
from typing import Optional
import threading

logger = logging.getLogger(__name__)


class AppWindow(ctk.CTk):
    """Main application window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        
        self.title("AutoIV-Ultimate - IV Data Analysis Suite")
        self.geometry("1200x800")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # State variables
        self.data_folder = ctk.StringVar(value="")
        self.output_folder = ctk.StringVar(value="output")
        self.scan_direction = ctk.StringVar(value="Reverse")
        self.theme = ctk.StringVar(value="dark")
        
        # Create UI
        self._create_ui()
        
        logger.info("AppWindow initialized")
    
    def _create_ui(self):
        """Create the complete UI layout."""
        # Create tab view
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tabview.add("Dashboard")
        self.tabview.add("Live Log")
        self.tabview.add("Plot Preview")
        
        # Populate tabs
        self._create_dashboard_tab()
        self._create_log_tab()
        self._create_preview_tab()
    
    def _create_dashboard_tab(self):
        """Create Dashboard tab with controls."""
        tab = self.tabview.tab("Dashboard")
        
        # Title
        title = ctk.CTkLabel(tab, text="AutoIV-Ultimate", font=("Arial", 24, "bold"))
        title.pack(pady=10)
        
        subtitle = ctk.CTkLabel(tab, text="Automated IV Characterization Analysis", font=("Arial", 14))
        subtitle.pack(pady=5)
        
        # Input folder selection
        folder_frame = ctk.CTkFrame(tab)
        folder_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(folder_frame, text="Data Folder:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        
        folder_input_frame = ctk.CTkFrame(folder_frame)
        folder_input_frame.pack(fill="x", padx=10, pady=5)
        
        self.folder_entry = ctk.CTkEntry(folder_input_frame, textvariable=self.data_folder, width=400)
        self.folder_entry.pack(side="left", padx=5)
        
        browse_btn = ctk.CTkButton(folder_input_frame, text="Browse", command=self._browse_folder, width=100)
        browse_btn.pack(side="left", padx=5)
        
        # Settings frame
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(settings_frame, text="Analysis Settings:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Scan direction
        scan_frame = ctk.CTkFrame(settings_frame)
        scan_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(scan_frame, text="Scan Direction:").pack(side="left", padx=10)
        scan_menu = ctk.CTkOptionMenu(scan_frame, variable=self.scan_direction, 
                                      values=["Reverse", "Forward", "All"])
        scan_menu.pack(side="left", padx=10)
        
        # Theme selection
        theme_frame = ctk.CTkFrame(settings_frame)
        theme_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(theme_frame, text="Plot Theme:").pack(side="left", padx=10)
        theme_menu = ctk.CTkOptionMenu(theme_frame, variable=self.theme, 
                                       values=["dark", "light"])
        theme_menu.pack(side="left", padx=10)
        
        # Action buttons
        button_frame = ctk.CTkFrame(tab)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        self.run_btn = ctk.CTkButton(button_frame, text="▶ Run Analysis", 
                                     command=self._run_analysis, 
                                     font=("Arial", 14, "bold"),
                                     height=40, width=200)
        self.run_btn.pack(side="left", padx=10)
        
        self.stop_btn = ctk.CTkButton(button_frame, text="⏹ Stop", 
                                      command=self._stop_analysis,
                                      font=("Arial", 14, "bold"),
                                      height=40, width=150,
                                      state="disabled")
        self.stop_btn.pack(side="left", padx=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(tab)
        self.progress.pack(fill="x", padx=20, pady=10)
        self.progress.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(tab, text="Ready", font=("Arial", 12))
        self.status_label.pack(pady=5)
    
    def _create_log_tab(self):
        """Create Live Log tab."""
        tab = self.tabview.tab("Live Log")
        
        # Title
        title = ctk.CTkLabel(tab, text="Live Analysis Log", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Log text box
        self.log_textbox = ctk.CTkTextbox(tab, font=("Consolas", 10))
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Clear button
        clear_btn = ctk.CTkButton(tab, text="Clear Log", command=self._clear_log, width=100)
        clear_btn.pack(pady=5)
    
    def _create_preview_tab(self):
        """Create Plot Preview tab."""
        tab = self.tabview.tab("Plot Preview")
        
        # Title
        title = ctk.CTkLabel(tab, text="Plot Preview", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Placeholder
        placeholder = ctk.CTkLabel(tab, text="Plots will appear here after analysis", 
                                  font=("Arial", 14))
        placeholder.pack(expand=True)
    
    # ================= EVENT HANDLERS =================
    
    def _browse_folder(self):
        """Open folder browser dialog."""
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            self.data_folder.set(folder)
            self._log(f"Selected folder: {folder}")
    
    def _run_analysis(self):
        """Run the analysis in a separate thread."""
        if not self.data_folder.get():
            self._log("ERROR: Please select a data folder first!")
            return
        
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.set(0)
        self.status_label.configure(text="Running analysis...")
        
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._analysis_worker, daemon=True)
        thread.start()
    
    def _analysis_worker(self):
        """Worker thread for analysis."""
        try:
            self._log("=" * 60)
            self._log("Starting AutoIV-Ultimate Analysis")
            self._log("=" * 60)
            
            # Import modules
            from autoiv_ultimate.core.data_loader import DataLoader
            from autoiv_ultimate.engines.statistics import StatisticsEngine
            from autoiv_ultimate.engines.physics import PhysicsEngine
            from autoiv_ultimate.visualization.plot_engine import PlotEngine
            from autoiv_ultimate.reporting.excel_builder import ExcelBuilder
            from autoiv_ultimate.reporting.word_builder import WordBuilder
            from autoiv_ultimate.reporting.ppt_builder import PowerPointBuilder
            
            # Configuration
            config = {
                'scan_direction': self.scan_direction.get(),
                'thresholds': {
                    'Eff_Min': 0.1,
                    'Voc_Min': 0.1,
                    'Jsc_Min': 0.1,
                    'FF_Min': 10.0,
                    'FF_Max': 90.0
                },
                'remove_duplicates': True,
                'outlier_removal': True
            }
            
            # Step 1: Load data
            self._log("\n[1/6] Loading data...")
            self.progress.set(0.1)
            data_loader = DataLoader(config)
            df, batch_order = data_loader.load_dataset(self.data_folder.get())
            self._log(f"✓ Loaded {len(df)} cells from {len(batch_order)} batches")
            
            # Step 2: Compute statistics
            self._log("\n[2/6] Computing statistics...")
            self.progress.set(0.3)
            stats_engine = StatisticsEngine(config)
            stats_df, champion_df, top_cells_df, yield_df, comparisons = stats_engine.compute_statistics(df, batch_order)
            self._log(f"✓ Statistics computed for {len(batch_order)} batches")
            
            # Step 3: Generate plots
            self._log("\n[3/6] Generating visualizations...")
            self.progress.set(0.5)
            group_colors = stats_engine.assign_colors(batch_order)
            plot_engine = PlotEngine(output_dir=self.output_folder.get(), theme=self.theme.get())
            img_paths = plot_engine.visualize_all(df, stats_df, champion_df, batch_order, group_colors)
            self._log(f"✓ Generated {len([p for p in img_paths.values() if p])} plots")
            
            # Step 4: Generate Excel report
            self._log("\n[4/6] Creating Excel report...")
            self.progress.set(0.7)
            excel_builder = ExcelBuilder(output_dir=self.output_folder.get())
            excel_path = excel_builder.create_report(stats_df, champion_df, top_cells_df, yield_df, df)
            self._log(f"✓ Excel report: {excel_path}")
            
            # Step 5: Generate Word report
            self._log("\n[5/6] Creating Word report...")
            self.progress.set(0.85)
            word_builder = WordBuilder(output_dir=self.output_folder.get())
            word_path = word_builder.create_report(stats_df, champion_df, img_paths)
            self._log(f"✓ Word report: {word_path}")
            
            # Step 6: Generate PowerPoint report
            self._log("\n[6/6] Creating PowerPoint report...")
            self.progress.set(0.95)
            ppt_builder = PowerPointBuilder(output_dir=self.output_folder.get())
            ppt_path = ppt_builder.create_report(stats_df, champion_df, img_paths)
            self._log(f"✓ PowerPoint report: {ppt_path}")
            
            # Complete
            self.progress.set(1.0)
            self._log("\n" + "=" * 60)
            self._log("✓ Analysis Complete!")
            self._log("=" * 60)
            self.status_label.configure(text="Analysis complete!")
            
        except Exception as e:
            self._log(f"\n❌ ERROR: {str(e)}")
            logger.exception("Analysis failed")
            self.status_label.configure(text="Analysis failed!")
        
        finally:
            self.run_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
    
    def _stop_analysis(self):
        """Stop the running analysis."""
        self._log("\n⏹ Analysis stopped by user")
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Stopped")
    
    def _clear_log(self):
        """Clear the log textbox."""
        self.log_textbox.delete("1.0", "end")
    
    def _log(self, message: str):
        """Add message to log textbox."""
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        logger.info(message)


def run():
    """Launch the GUI application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = AppWindow()
    app.mainloop()


if __name__ == "__main__":
    run()
