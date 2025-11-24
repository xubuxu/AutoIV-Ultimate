"""
AutoIV-Ultimate Setup Script
"""
from setuptools import setup, find_packages

setup(
    name="autoiv-ultimate",
    version="1.0.0",
    description="Automated IV Characterization Analysis Suite",
    author="AutoIV Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "openpyxl>=3.0.0",
        "python-docx>=0.8.11",
        "python-pptx>=0.6.21",
        "customtkinter>=5.0.0",
        "darkdetect>=0.8.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "autoiv-ultimate=autoiv_ultimate.ui.app_window:run",
        ],
    },
)
