"scr/path_config.py"

"""
    this file will contains  all  the essentail paths needed in this project 
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITED_DATA_DIR = RAW_DATA_DIR / "split"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KAGGLE_DATA_PATH = RAW_DATA_DIR / "kaggle" / "creditcard.csv"
EXP_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = PROJECT_ROOT /"reports"/"figures"
