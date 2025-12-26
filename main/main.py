#!/usr/bin/env python3
"""
Political Stability Prediction - Main Entry Point
==================================================

This is the main entry point for the entire project workflow.
It provides an interactive menu to execute each step of the ML pipeline:

0. Check Environment       - Verify dependencies and data
1. Run Data Preparation    - Load, clean, and prepare datasets
2. Train Model             - Train ML/econometric models
3. Test Model              - Test trained models
4. Evaluate Saved Models   - Compare all saved models
5. Run Visualization       - Generate plots and charts
6. Show Dashboard Link     - Display Streamlit dashboard URL
7. Test Coverage           - Run test coverage analysis
"""

import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

# Add project root to sys.path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# LOGGING CONFIGURATION (MUST BE DONE FIRST, BEFORE ALL IMPORTS)
# ============================================================================
# - Centralized logging for entire project
# - All modules (src/models.py, src/data_loader.py, etc.) will use this config
# - Prevents duplicate logging configurations
# ============================================================================

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Configure logging BEFORE importing any project modules (console only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # Console handler (shows logs in terminal)
        logging.StreamHandler()
    ],
)

# ============================================================================
# REPRODUCIBILITY: Fix all random seeds for consistent results
# ============================================================================
# This ensures that running the code multiple times produces identical results,
# which is essential for scientific reproducibility and model comparison.
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# ============================================================================


# ============================================================================
# SESSION TRACKING
# ============================================================================

# Track if training has been completed in this session
TRAINING_COMPLETED_THIS_SESSION = False

# Store trained models and data from current session (in-memory, no .pkl files)
SESSION_MODELS_DATA = None
SESSION_TRAIN_DATA = None
SESSION_TEST_DATA = None

# Track completion status of each workflow step
WORKFLOW_COMPLETION = {
    "check_environment": False,
    "data_preparation": False,
    "train_model": False,
    "test_model": False,
    "evaluate_models": False,
    "visualization": False,
    "dashboard": False,
    "test_coverage": False,
}


# ============================================================================
# TERMINAL FORMATTING & COLORS
# ============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str, subtitle: str = ""):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{title.center(80)}{Colors.RESET}")
    if subtitle:
        print(f"{Colors.CYAN}{subtitle.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'' * 78}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE} {title}{Colors.RESET}{' ' * (75 - len(title))}{Colors.BOLD}{Colors.BRIGHT_CYAN}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'' * 78}{Colors.RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.BRIGHT_GREEN} {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.BRIGHT_RED} {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.BRIGHT_YELLOW} {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.CYAN}i {message}{Colors.RESET}")


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================


def get_project_paths():
    """Get all important project paths."""
    # Since main.py is now in main/ folder, go up one level to get project root
    project_root = Path(__file__).parent.parent

    paths = {
        "root": project_root,
        "data_raw": project_root / "data" / "raw",
        "data_processed": project_root / "data" / "processed",
        "results": project_root / "results",
        "figures": project_root / "results" / "figures",
        "src": project_root / "src",
    }

    return paths


# ============================================================================
# ACTION 0: CHECK ENVIRONMENT
# ============================================================================


def check_environment():
    """
    Check that all dependencies and data files are available.

    Verifies:
    - Python version
    - Required packages
    - Data directories
    - Raw data files
    """
    print_section("ACTION 0: CHECK ENVIRONMENT")

    all_ok = True

    # 1. Check Python version
    print_info("Checking Python version...")
    python_version = sys.version_info
    if python_version >= (3, 8):
        print_success(
            f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
    else:
        print_error(
            f"Python {python_version.major}.{python_version.minor} (requires >= 3.8)"
        )
        all_ok = False

    # 2. Check required packages
    print_info("\nChecking required packages...")
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "linearmodels",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "streamlit",
    ]

    for package in required_packages:
        try:
            pkg_version = version(package)
            print_success(f"{package} ({pkg_version})")
        except PackageNotFoundError:
            print_error(f"{package} - NOT INSTALLED")
            all_ok = False

    # 3. Check directories
    print_info("\nChecking project directories...")
    paths = get_project_paths()

    for name, path in paths.items():
        if path.exists():
            print_success(f"{name}: {path}")
        else:
            print_warning(f"{name}: {path} - MISSING (will be created)")
            path.mkdir(parents=True, exist_ok=True)

    # 4. Check raw data files (accept both .csv and .numbers)
    print_info("\nChecking raw data files...")
    data_raw = paths["data_raw"]

    # Check for files - accept multiple formats
    # Each entry: (base_name, list_of_extensions)
    expected_files = [
        ("GDP per capita", [".numbers", ".csv"]),  # GDP per capita.numbers
        ("UNEMPLOYMENT_TOTAL", [".numbers", ".csv"]),  # UNEMPLOYMENT_TOTAL.numbers
        ("inflation consumer", [".numbers", ".csv"]),  # inflation consumer.numbers
        ("hdi_data", [".xlsx"]),  # hdi_data.xlsx
    ]

    for base_name, extensions in expected_files:
        found = False
        for ext in extensions:
            filepath = data_raw / f"{base_name}{ext}"
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print_success(f"{base_name}{ext} ({size_mb:.2f} MB)")
                found = True
                break

        if not found:
            print_error(f"{base_name} - NOT FOUND (checked: {', '.join(extensions)})")
            all_ok = False

    # Summary
    print()
    if all_ok:
        print_success("Environment check: ALL OK")

        # Mark as completed
        global WORKFLOW_COMPLETION
        WORKFLOW_COMPLETION["check_environment"] = True
    else:
        print_error("Environment check: SOME ISSUES FOUND")
        print()
        print(f"{Colors.CYAN}To fix missing dependencies:{Colors.RESET}")
        print()
        print(f"  {Colors.BOLD}Option 1: Using Conda (recommended){Colors.RESET}")
        print(f"    {Colors.DIM}conda env create -f environment.yml{Colors.RESET}")
        print(
            f"    {Colors.DIM}conda activate political-stability-prediction{Colors.RESET}"
        )
        print()
        print(f"  {Colors.BOLD}Option 2: Using pip{Colors.RESET}")
        print(f"    {Colors.DIM}pip install -r requirements.txt{Colors.RESET}")
        print()
        print(f"  {Colors.BOLD}For missing data files:{Colors.RESET}")
        print(
            f"    {Colors.DIM}Download required datasets to: {paths['data_raw']}{Colors.RESET}"
        )

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 1: RUN DATA PREPARATION
# ============================================================================


def run_data_preparation():
    """
    Load raw data, clean, merge, and save processed datasets.

    Steps:
    1. Load raw data from CSV/Excel files
    2. Clean and validate data
    3. Merge all indicators
    4. Save processed data
    """
    print_section("ACTION 1: RUN DATA PREPARATION")

    try:
        # Import data loader module
        import pandas as pd

        from src.data_loader import load_data

        paths = get_project_paths()

        print_info("Step 1/4: Loading raw data files...")
        print(
            f"{Colors.DIM}        Reading GDP, unemployment, inflation, HDI data...{Colors.RESET}"
        )

        # Load and process data
        data_dict = load_data(
            data_path=paths["data_raw"],
            target="political_stability",
            train_end_year=2017,
        )

        print_success("Raw data loaded successfully")

        print_info("Step 2/4: Cleaning and validating data...")
        print(
            f"{Colors.DIM}        Removing outliers, handling missing values...{Colors.RESET}"
        )
        print_success("Data cleaning completed")

        print_info("Step 3/4: Merging indicators...")
        print(
            f"{Colors.DIM}        Combining economic, social, and political indicators...{Colors.RESET}"
        )
        print_success("Data merge completed")

        print_info("Step 4/4: Saving processed datasets...")

        # Save processed data
        processed_dir = paths["data_processed"]
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Combine train and test data
        X_train = data_dict["X_train"]
        X_test = data_dict["X_test"]
        y_train = data_dict["y_train"]
        y_test = data_dict["y_test"]

        # Save train data
        train_data = X_train.copy()
        train_data["political_stability"] = y_train
        train_file = processed_dir / "train_data.csv"
        train_data.to_csv(train_file, index=True)

        # Save test data
        test_data = X_test.copy()
        test_data["political_stability"] = y_test
        test_file = processed_dir / "test_data.csv"
        test_data.to_csv(test_file, index=True)

        # Save full data
        full_data = pd.concat([train_data, test_data])
        full_file = processed_dir / "full_data.csv"
        full_data.to_csv(full_file, index=True)

        print_success("Processed data saved")

        print()
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_GREEN} DATA PREPARATION COMPLETED{Colors.RESET}"
        )
        print()
        print(f"{Colors.CYAN}Output files:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} train_data.csv    {Colors.DIM}({train_file.stat().st_size / 1024:.1f} KB, {len(train_data)} samples){Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} test_data.csv     {Colors.DIM}({test_file.stat().st_size / 1024:.1f} KB, {len(test_data)} samples){Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} full_data.csv     {Colors.DIM}({full_file.stat().st_size / 1024:.1f} KB, {len(full_data)} samples){Colors.RESET}"
        )
        print()
        print(f"{Colors.CYAN}Dataset info:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Training samples:   {Colors.BOLD}{len(train_data)}{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Test samples:       {Colors.BOLD}{len(test_data)}{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Features:           {Colors.BOLD}{len(X_train.columns)}{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Target variable:    {Colors.BOLD}political_stability{Colors.RESET}"
        )

    except ImportError as e:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'' * 78}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_RED} MODULE ERROR{Colors.RESET}{' ' * 62}{Colors.BOLD}{Colors.BRIGHT_RED}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'' * 78}{Colors.RESET}")
        print()
        print(f"{Colors.YELLOW}Unable to import required module:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} {Colors.DIM}{str(e)}{Colors.RESET}"
        )
        print()
        print(f"{Colors.CYAN}Possible solutions:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}1.{Colors.RESET} Ensure all dependencies are installed:"
        )
        print(
            f"     {Colors.DIM}python3 -m pip install numpy pandas scikit-learn{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}2.{Colors.RESET} Check that the project structure is intact"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}3.{Colors.RESET} Restart your terminal and try again"
        )
        print()
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    except FileNotFoundError as e:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'' * 78}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_RED} FILE NOT FOUND{Colors.RESET}{' ' * 59}{Colors.BOLD}{Colors.BRIGHT_RED}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'' * 78}{Colors.RESET}")
        print()
        print(f"{Colors.YELLOW}Required data file is missing:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} {Colors.DIM}{str(e)}{Colors.RESET}"
        )
        print()
        print(f"{Colors.CYAN}Solution:{Colors.RESET}")
        print(
            f"  Run {Colors.BOLD}[0] Check Environment{Colors.RESET} to verify all data files"
        )
        print()
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    except Exception as e:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'' * 78}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_RED} DATA PREPARATION FAILED{Colors.RESET}{' ' * 51}{Colors.BOLD}{Colors.BRIGHT_RED}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'' * 78}{Colors.RESET}")
        print()
        print(f"{Colors.YELLOW}An unexpected error occurred:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} {Colors.DIM}{str(e)}{Colors.RESET}"
        )
        print()
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    # Mark as completed (only if no exceptions occurred)
    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["data_preparation"] = True

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 2: TRAIN MODEL
# ============================================================================


def save_detailed_results(
    filepath, model_name, model, metrics, X_train, y_train, X_test, y_test, elapsed
):
    """
    Save detailed model results to text file including overfitting metrics.

    Parameters
    ----------
    filepath : Path
        Output file path
    model_name : str
        Name of the model
    model : object
        Trained model instance
    metrics : dict
        Evaluation metrics including overfitting diagnostics
    X_train, y_train : DataFrame, Series
        Training data
    X_test, y_test : DataFrame, Series
        Test data
    elapsed : float
        Training time in seconds
    """
    from pathlib import Path

    import numpy as np

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{model_name.upper()} RESULTS - POLITICAL STABILITY PREDICTION\n")
        f.write("=" * 100 + "\n\n")

        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Target: political_stability\n")
        f.write(f"Predictors: {', '.join(X_train.columns)}\n")
        f.write(f"Train period: 1996-2017\n")
        f.write(f"Test period: 2018-2023\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n")
        f.write(f"Training time: {elapsed:.2f}s\n\n")

        # Best hyperparameters
        if hasattr(model, "best_params_") and model.best_params_:
            f.write("BEST HYPERPARAMETERS\n")
            f.write("-" * 100 + "\n")
            for param, value in sorted(model.best_params_.items()):
                f.write(f"{param}: {value}\n")
            f.write("\n")

        # Overfitting diagnostics
        if metrics.get("train_score") is not None:
            f.write("OVERFITTING DIAGNOSTICS\n")
            f.write("-" * 100 + "\n")
            f.write(f"Mean Train R^2: {metrics['train_score']:.4f}\n")
            f.write(f"Mean CV R^2: {metrics['cv_score']:.4f}\n")
            f.write(f"Train-CV Gap: {metrics['overfitting_gap']:.4f}\n")

            if metrics["has_overfitting"]:
                f.write(f"Overfitting Status: [WARNING]  YES (gap > 0.1)\n")
            else:
                f.write(f"Overfitting Status: OK NO (gap < 0.1)\n")
            f.write("\n")

        # Cross-validation results
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-" * 100 + "\n")
        if metrics.get("cv_score") is not None:
            f.write(f"Best CV R^2: {metrics['cv_score']:.4f}\n")
        f.write("\n")

        # Test set performance
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 100 + "\n")
        f.write(f"Test R^2: {metrics['r2']:.4f}\n")
        f.write(f"Test Adjusted R^2: {metrics['adj_r2']:.4f}\n")
        f.write(f"Test RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"Test MAE: {metrics['mae']:.4f}\n")
        f.write(f"Number of samples: {metrics.get('n_samples', 'N/A')}\n")
        f.write(f"Number of features: {metrics.get('n_features', 'N/A')}\n\n")

        # F-statistic (model significance)
        if metrics.get("f_stat") is not None:
            f.write("MODEL SIGNIFICANCE (F-STATISTIC)\n")
            f.write("-" * 100 + "\n")
            f.write(f"F-statistic: {metrics['f_stat']:.4f}\n")
            f.write(f"F p-value: {metrics['f_pvalue']:.6f}\n")
            if metrics["f_pvalue"] < 0.001:
                f.write(f"Significance: *** (p < 0.001) - Highly significant\n")
            elif metrics["f_pvalue"] < 0.01:
                f.write(f"Significance: ** (p < 0.01) - Very significant\n")
            elif metrics["f_pvalue"] < 0.05:
                f.write(f"Significance: * (p < 0.05) - Significant\n")
            else:
                f.write(f"Significance: Not significant (p >= 0.05)\n")
            f.write("\n")

        # Feature importance (if available)
        if hasattr(model.model, "feature_importances_"):
            importances = model.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            f.write("FEATURE IMPORTANCE\n")
            f.write("-" * 100 + "\n")
            for i in indices:
                f.write(f"{X_train.columns[i]:30s}: {importances[i]:.4f}\n")
            f.write("\n")

    print(f"  {Colors.DIM}-> Saved detailed results to {filepath.name}{Colors.RESET}")


def train_all_ml_models():
    """Train all models: 7 ML models + Dynamic Panel Analysis."""
    import time

    import pandas as pd

    from src.models import (
        ElasticNetPredictor,
        GradientBoostingPredictor,
        KNNPredictor,
        MLPPredictor,
        RandomForestPredictor,
        SVRPredictor,
        XGBoostPredictor,
    )

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'TRAINING ALL MODELS - COMPLETE BENCHMARK'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")
    print_info("Training 7 ML models + Dynamic Panel Analysis")
    print(f"{Colors.DIM}Estimated time: 5-10 minutes{Colors.RESET}\n")

    # Get paths
    paths = get_project_paths()
    processed_dir = paths["data_processed"]
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    print_info("Step 1/6: Loading processed data...")
    try:
        train_data = pd.read_csv(processed_dir / "train_data.csv", index_col=[0, 2])
        test_data = pd.read_csv(processed_dir / "test_data.csv", index_col=[0, 2])

        # Drop Country Code column if it exists (it's not a feature)
        if "Country Code" in train_data.columns:
            train_data = train_data.drop("Country Code", axis=1)
        if "Country Code" in test_data.columns:
            test_data = test_data.drop("Country Code", axis=1)

        X_train = train_data.drop("political_stability", axis=1)
        y_train = train_data["political_stability"]
        X_test = test_data.drop("political_stability", axis=1)
        y_test = test_data["political_stability"]

        # Display train/test split information
        train_years = train_data.index.get_level_values(1).unique()
        test_years = test_data.index.get_level_values(1).unique()

        print_success(
            f"Loaded {len(train_data)} training samples, {len(test_data)} test samples"
        )
        print(
            f"{Colors.DIM}        Train years: {int(train_years.min())}-{int(train_years.max())} | Test years: {int(test_years.min())}-{int(test_years.max())}{Colors.RESET}"
        )
    except Exception as e:
        print_error(f"Failed to load data: {str(e)}")
        print_info("Please run [1] Run Data Preparation first")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    # Define models to train
    models = [
        ("Random Forest", RandomForestPredictor()),
        ("XGBoost", XGBoostPredictor()),
        ("Gradient Boosting", GradientBoostingPredictor()),
        ("Elastic Net", ElasticNetPredictor()),
        ("SVR", SVRPredictor()),
        ("KNN", KNNPredictor()),
        ("MLP", MLPPredictor()),
    ]

    results = []

    print()
    print_info(f"Step 2/6: Training {len(models)} ML models...")
    print(f"{Colors.DIM}This may take several minutes...{Colors.RESET}\n")

    # Train each model
    for i, (model_name, model) in enumerate(models, 1):
        print(f"{Colors.CYAN}{'' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_WHITE}[{i}/{len(models)}] {model_name}{Colors.RESET}"
        )
        print(f"{Colors.CYAN}{'' * 80}{Colors.RESET}")

        start_time = time.time()

        try:
            # Train
            print(f"  {Colors.DIM}-> Training...{Colors.RESET}")
            model.fit(X_train, y_train)

            # Evaluate
            print(f"  {Colors.DIM}-> Evaluating...{Colors.RESET}")
            metrics = model.evaluate(X_test, y_test)

            elapsed = time.time() - start_time

            # Display results with new metrics (Adjusted R², F-statistic)
            print(
                f"  {Colors.BRIGHT_GREEN}OK Test R^2 = {metrics['r2']:.4f} | Adj R^2 = {metrics['adj_r2']:.4f} | MAE = {metrics['mae']:.4f} | Time = {elapsed:.1f}s{Colors.RESET}"
            )

            # Display F-statistic (model significance)
            if metrics.get("f_stat") is not None:
                f_stat = metrics["f_stat"]
                f_pvalue = metrics["f_pvalue"]
                if f_pvalue < 0.001:
                    sig_str = "p < 0.001 ***"
                    color = Colors.GREEN
                elif f_pvalue < 0.01:
                    sig_str = f"p = {f_pvalue:.3f} **"
                    color = Colors.GREEN
                elif f_pvalue < 0.05:
                    sig_str = f"p = {f_pvalue:.3f} *"
                    color = Colors.GREEN
                else:
                    sig_str = f"p = {f_pvalue:.3f}"
                    color = Colors.YELLOW

                print(
                    f"  {Colors.DIM}F-statistic:{Colors.RESET} {f_stat:.2f} ({color}{sig_str}{Colors.RESET})"
                )

            # Display overfitting diagnostics
            if metrics.get("train_score") is not None:
                train_score = metrics["train_score"]
                cv_score = metrics["cv_score"]
                gap = metrics["overfitting_gap"]
                has_overfit = metrics["has_overfitting"]

                print(f"  {Colors.DIM}Overfitting Diagnostics:{Colors.RESET}")
                print(
                    f"    Train R^2: {train_score:.4f} | CV R^2: {cv_score:.4f} | Gap: {gap:.4f}"
                )

                if has_overfit:
                    print(
                        f"    {Colors.YELLOW}[WARNING]  Overfitting detected (gap > 0.1){Colors.RESET}"
                    )
                else:
                    print(
                        f"    {Colors.GREEN}OK No significant overfitting{Colors.RESET}"
                    )

            # Store results with trained model and predictions
            y_pred = model.predict(X_test)
            results.append(
                {
                    "model": model_name,
                    "model_obj": model,  # Store trained model object
                    "y_pred": y_pred,  # Store predictions
                    "y_test": y_test,  # Store test targets
                    "X_test": X_test,  # Store test features
                    "X_train": X_train,  # Store training features (for learning curves, CV)
                    "r2": metrics["r2"],
                    "adj_r2": metrics["adj_r2"],
                    "f_stat": metrics["f_stat"],
                    "f_pvalue": metrics["f_pvalue"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "time": elapsed,
                    "train_score": metrics.get("train_score"),
                    "cv_score": metrics.get("cv_score"),
                    "overfitting_gap": metrics.get("overfitting_gap"),
                    "n_samples": metrics.get("n_samples"),
                    "n_features": metrics.get("n_features"),
                }
            )

            # Save detailed results file
            results_file = (
                results_dir / f"ml_{model_name.lower().replace(' ', '_')}_results.txt"
            )
            save_detailed_results(
                results_file,
                model_name,
                model,
                metrics,
                X_train,
                y_train,
                X_test,
                y_test,
                elapsed,
            )

        except Exception as e:
            print(f"  {Colors.BRIGHT_RED} Training failed: {str(e)}{Colors.RESET}")

        print()

    # Save ML results to CSV
    if results:
        results_dir = paths["results"]
        results_dir.mkdir(parents=True, exist_ok=True)

        # Use fixed filename (overwrites previous results)
        results_file = results_dir / "benchmark_results.csv"

        # Sort by R^2 score
        results_df = pd.DataFrame(results).sort_values("r2", ascending=False)

        # Add metadata
        results_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_df["train_samples"] = len(X_train)
        results_df["test_samples"] = len(X_test)

        # Save to CSV
        results_df.to_csv(results_file, index=False)

    # Now train Dynamic Panel Analysis
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'STEP 2/2: DYNAMIC PANEL ANALYSIS'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

    # Call the panel training function and get its metrics
    panel_metrics = train_dynamic_panel(skip_input=True)

    # UNIFIED SUMMARY - Display both ML and Panel results together
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'ALL MODELS TRAINED - UNIFIED RESULTS'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

    # Display ML models ranking
    if results:
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}Machine Learning Models:{Colors.RESET}\n"
        )
        results_df = pd.DataFrame(results).sort_values("r2", ascending=False)

        for idx, row in results_df.iterrows():
            rank = idx + 1
            print(
                f"  {Colors.BRIGHT_WHITE}{rank}.{Colors.RESET} {row['model']:20s} "
                f"R^2={Colors.BOLD}{row['r2']:.4f}{Colors.RESET}  "
                f"MAE={row['mae']:.4f}  "
                f"RMSE={row['rmse']:.4f}  "
                f"({row['time']:.1f}s)"
            )

    # Display Panel Analysis results
    if panel_metrics:
        print()
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}Dynamic Panel Analysis:{Colors.RESET}\n"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}6.{Colors.RESET} {'Dynamic Panel (FE)':20s} "
            f"R^2={Colors.BOLD}{panel_metrics['r2']:.4f}{Colors.RESET}  "
            f"RMSE={panel_metrics['rmse']:.4f}      "
            f"({panel_metrics['time']:.1f}s)"
        )

    # Best model comparison
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}Summary:{Colors.RESET}")
    if results:
        best_ml_model = results_df.iloc[0]
        print(
            f"  {Colors.BRIGHT_GREEN}OK{Colors.RESET} Best ML Model:  {best_ml_model['model']} (R^2 = {best_ml_model['r2']:.4f})"
        )
    # Panel Model line removed as per user request

    print()
    print(f"{Colors.CYAN}Results saved to:{Colors.RESET}")
    print(
        f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} results/       {Colors.DIM}-> CSV files and panel results{Colors.RESET}"
    )

    # Generate ML vs Benchmark comparison visualization
    if results and panel_metrics:
        print()
        print_info("Generating ML vs Benchmark comparison visualization...")
        try:
            from src.evaluation import generate_ml_vs_benchmark_comparison

            figures_dir = paths.get("figures", paths["results"] / "figures")
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Pass trained models directly (no .pkl loading)
            viz_file = generate_ml_vs_benchmark_comparison(
                ml_results=results,  # Contains model_obj, y_pred, etc.
                panel_metrics=panel_metrics,
                output_dir=figures_dir,
            )
            print_success(f"Visualization saved: {viz_file.name}")
            print(f"  {Colors.DIM}-> {viz_file}{Colors.RESET}")
        except Exception as e:
            print_error(f"Failed to generate visualization: {str(e)}")

    # Mark training as completed in this session and store models in memory
    global TRAINING_COMPLETED_THIS_SESSION, WORKFLOW_COMPLETION, SESSION_MODELS_DATA, SESSION_TRAIN_DATA, SESSION_TEST_DATA
    TRAINING_COMPLETED_THIS_SESSION = True
    WORKFLOW_COMPLETION["train_model"] = True

    # Store models and data in session for visualizations (NO .pkl files)
    SESSION_MODELS_DATA = results
    SESSION_TRAIN_DATA = train_data
    SESSION_TEST_DATA = test_data

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def train_dynamic_panel(skip_input=False):
    """Train Dynamic Panel Analysis models.

    Returns:
        dict: Panel metrics (r2, rmse, time) if successful, None otherwise
    """
    import time

    import pandas as pd

    from src.models import PanelAnalyzer

    if not skip_input:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'DYNAMIC PANEL ANALYSIS'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

    # Get paths
    paths = get_project_paths()
    processed_dir = paths["data_processed"]
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    # For panel analysis, always reload from raw data to ensure proper structure
    print_info("Loading data from raw sources...")
    try:
        from src.data_loader import load_data

        data_dict = load_data(
            data_path=paths["root"] / "data" / "raw",
            target="political_stability",
            train_end_year=2017,
        )

        # Get the FULL dataframe before train/test split (has Country Name and Year)
        df_train = data_dict["df_train"].copy()
        df_test = data_dict["df_test"].copy()

        # Combine train and test
        full_data = pd.concat([df_train, df_test])

        # Reset index to make Country Name and Year regular columns
        # Handle both 2-level and 3-level MultiIndex (with or without Country Code)
        if isinstance(df_train.index, pd.MultiIndex):
            df_train = df_train.reset_index()
        if isinstance(df_test.index, pd.MultiIndex):
            df_test = df_test.reset_index()
        if isinstance(full_data.index, pd.MultiIndex):
            full_data = full_data.reset_index()

        # Check if we have the required columns
        if "Country Name" not in full_data.columns or "Year" not in full_data.columns:
            raise ValueError("Missing required columns: Country Name or Year")

        print_success(f"Loaded {len(full_data)} total observations")
        print(
            f"  {Colors.DIM}-> Training period: {int(df_train['Year'].min())}-{int(df_train['Year'].max())}{Colors.RESET}"
        )
        print(
            f"  {Colors.DIM}-> Test period: {int(df_test['Year'].min())}-{int(df_test['Year'].max())}{Colors.RESET}"
        )
    except Exception as e:
        print_error(f"Failed to load data: {str(e)}")
        print_info("Please run [1] Run Data Preparation first")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    # Save temporary CSV for PanelAnalyzer
    temp_csv = paths["data_processed"] / "temp_panel_data.csv"
    full_data.to_csv(temp_csv, index=False)

    print()
    print_info("Initializing Panel Analyzer...")

    start_time = time.time()

    try:
        # Get predictors (all columns except target, entity and time identifiers)
        predictors = [
            col
            for col in full_data.columns
            if col
            not in ["Country Name", "Country Code", "Year", "political_stability"]
        ]

        # Initialize analyzer
        analyzer = PanelAnalyzer(
            data_path=temp_csv, target="political_stability", predictors=predictors
        )

        # Load and prepare data
        print(f"  {Colors.DIM}-> Preparing panel structure...{Colors.RESET}")
        analyzer.load_data()
        analyzer.prepare_panel_structure()

        # Train/test split
        # Use 2017 as train end year (standard for this project)
        train_end_year = 2017
        print(
            f"  {Colors.DIM}-> Splitting data (train end: {train_end_year})...{Colors.RESET}"
        )
        analyzer.train_test_split(
            test_years=None, train_end_year=train_end_year, test_ratio=0.2
        )

        # Fit Dynamic Panel model
        print()
        print_info("Fitting Dynamic Panel model...")

        print(f"  {Colors.DIM}-> Dynamic Panel (FE)...{Colors.RESET}")
        analyzer.fit_dynamic_panel(lags=1, use_train_only=True)

        # Evaluate on test set
        print()
        print_info("Evaluating on test set...")
        metrics = analyzer.evaluate_on_test()

        elapsed = time.time() - start_time

        # Save results (fixed filename, overwrites previous)
        from datetime import datetime

        results_file = results_dir / "panel_analysis.txt"

        with open(results_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("DYNAMIC PANEL ANALYSIS RESULTS (Fixed Effects)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test R^2: {metrics['r2']:.4f}\n")
            f.write(f"Test RMSE: {metrics['rmse']:.4f}\n\n")
            f.write("DYNAMIC PANEL MODEL\n")
            f.write("-" * 80 + "\n")
            f.write(str(analyzer.dynamic_results.summary))

        # Show results only if not skipping input (standalone mode)
        if not skip_input:
            print()
            print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
            print(
                f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'PANEL ANALYSIS COMPLETE'.center(80)}{Colors.RESET}"
            )
            print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

            print(f"{Colors.CYAN}Test Set Performance:{Colors.RESET}\n")
            print(
                f"  {Colors.BRIGHT_WHITE}R^2 Score:{Colors.RESET}  {Colors.BOLD}{metrics['r2']:.4f}{Colors.RESET}"
            )
            print(
                f"  {Colors.BRIGHT_WHITE}RMSE:{Colors.RESET}      {metrics['rmse']:.4f}"
            )

            print()
            print(f"  {Colors.BRIGHT_WHITE}Time:{Colors.RESET}      {elapsed:.1f}s")

            print()
            print(f"{Colors.CYAN}Results saved to:{Colors.RESET}")
            print(f"  {Colors.DIM}-> {results_file}{Colors.RESET}")

        # Cleanup
        if temp_csv.exists():
            temp_csv.unlink()

        # Return metrics for unified summary
        return {"r2": metrics["r2"], "rmse": metrics["rmse"], "time": elapsed}

    except Exception as e:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{'PANEL ANALYSIS FAILED'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'=' * 80}{Colors.RESET}\n")
        print_error(f"Error: {str(e)}")

        # Cleanup on error
        if temp_csv.exists():
            temp_csv.unlink()

        return None

    finally:
        if not skip_input:
            # Mark training as completed in this session (standalone panel training)
            global TRAINING_COMPLETED_THIS_SESSION
            TRAINING_COMPLETED_THIS_SESSION = True
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def train_single_model(model_name, ModelClass):
    """Train a single ML model."""
    import time

    import pandas as pd

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{f'TRAINING {model_name.upper()}'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

    # Get paths
    paths = get_project_paths()
    processed_dir = paths["data_processed"]

    # Load processed data
    print_info("Loading processed data...")
    try:
        train_data = pd.read_csv(processed_dir / "train_data.csv", index_col=[0, 2])
        test_data = pd.read_csv(processed_dir / "test_data.csv", index_col=[0, 2])

        # Drop Country Code column if it exists (it's not a feature)
        if "Country Code" in train_data.columns:
            train_data = train_data.drop("Country Code", axis=1)
        if "Country Code" in test_data.columns:
            test_data = test_data.drop("Country Code", axis=1)

        X_train = train_data.drop("political_stability", axis=1)
        y_train = train_data["political_stability"]
        X_test = test_data.drop("political_stability", axis=1)
        y_test = test_data["political_stability"]

        print_success(
            f"Loaded {len(train_data)} training samples, {len(test_data)} test samples"
        )
    except Exception as e:
        print_error(f"Failed to load data: {str(e)}")
        print_info("Please run [1] Run Data Preparation first")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    print()
    print_info(f"Training {model_name}...")
    print(f"{Colors.DIM}This may take a few minutes...{Colors.RESET}\n")

    start_time = time.time()

    try:
        # Initialize model
        model = ModelClass()

        # Train
        print(f"  {Colors.DIM}-> Training model...{Colors.RESET}")
        model.fit(X_train, y_train)

        # Evaluate
        print(f"  {Colors.DIM}-> Evaluating on test set...{Colors.RESET}")
        metrics = model.evaluate(X_test, y_test)

        elapsed = time.time() - start_time

        # Show results
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'TRAINING COMPLETE'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

        print(f"{Colors.CYAN}Performance Metrics:{Colors.RESET}\n")
        print(
            f"  {Colors.BRIGHT_WHITE}R^2 Score:{Colors.RESET}  {Colors.BOLD}{metrics['r2']:.4f}{Colors.RESET}"
        )
        print(f"  {Colors.BRIGHT_WHITE}MAE:{Colors.RESET}       {metrics['mae']:.4f}")
        print(f"  {Colors.BRIGHT_WHITE}RMSE:{Colors.RESET}      {metrics['rmse']:.4f}")
        print(f"  {Colors.BRIGHT_WHITE}Time:{Colors.RESET}      {elapsed:.1f}s")

    except Exception as e:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{'TRAINING FAILED'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'=' * 80}{Colors.RESET}\n")
        print_error(f"Error: {str(e)}")

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def train_model():
    """
    Train ML or econometric models.

    User can choose:
    - All models (7 ML + Dynamic Panel)
    - Individual ML model (RF, XGBoost, SVR, KNN, MLP)
    - Dynamic Panel Analysis only
    """
    print_section("ACTION 2: TRAIN MODEL")

    # Check workflow: Must have prepared data first
    global WORKFLOW_COMPLETION
    if not WORKFLOW_COMPLETION["data_preparation"]:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{'WORKFLOW ERROR: DATA PREPARATION REQUIRED'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}\n")

        print_error("You must prepare data first before training models!")
        print()
        print(
            f"{Colors.CYAN}Training requires processed data from the data preparation step.{Colors.RESET}"
        )
        print()
        print(f"{Colors.BRIGHT_WHITE}Required workflow:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_GREEN}1.{Colors.RESET} Run {Colors.BOLD}[1] Data Preparation{Colors.RESET} first"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}2.{Colors.RESET} Then run {Colors.BOLD}[2] Train Model{Colors.RESET}"
        )
        print()
        print(
            f"{Colors.DIM}Note: If you restart the program, you must re-run data preparation.{Colors.RESET}"
        )

        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Choose model to train:{Colors.RESET}")
    print(f"{Colors.DIM}{'' * 80}{Colors.RESET}\n")

    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[1]{Colors.RESET}  ALL MODELS (ML + Panel)     {Colors.DIM}->{Colors.RESET}  Complete benchmark (8 models)"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[2]{Colors.RESET}  Dynamic Panel Only          {Colors.DIM}->{Colors.RESET}  Econometric panel models"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[3]{Colors.RESET}  Random Forest               {Colors.DIM}->{Colors.RESET}  Train RF model"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[4]{Colors.RESET}  XGBoost                     {Colors.DIM}->{Colors.RESET}  Train XGBoost model"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[5]{Colors.RESET}  Gradient Boosting           {Colors.DIM}->{Colors.RESET}  Train GB model"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[6]{Colors.RESET}  Elastic Net                 {Colors.DIM}->{Colors.RESET}  Train Elastic Net"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[7]{Colors.RESET}  SVR                         {Colors.DIM}->{Colors.RESET}  Support Vector Regression"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[8]{Colors.RESET}  KNN                         {Colors.DIM}->{Colors.RESET}  K-Nearest Neighbors"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[9]{Colors.RESET}  MLP                         {Colors.DIM}->{Colors.RESET}  Neural Network"
    )

    print(f"\n{Colors.DIM}{'' * 80}{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_RED}[0]{Colors.RESET}  Back to main menu")

    choice = input(f"\n{Colors.BOLD}Enter choice [0-9]: {Colors.RESET}").strip()

    if choice == "0":
        return

    if choice == "1":
        # Train ALL models (7 ML + Dynamic Panel)
        train_all_ml_models()
    elif choice == "2":
        # Train Dynamic Panel Analysis only
        train_dynamic_panel()
    elif choice in ["3", "4", "5", "6", "7", "8", "9"]:
        # Train individual model
        from src.models import (
            ElasticNetPredictor,
            GradientBoostingPredictor,
            KNNPredictor,
            MLPPredictor,
            RandomForestPredictor,
            SVRPredictor,
            XGBoostPredictor,
        )

        model_map = {
            "3": ("Random Forest", RandomForestPredictor),
            "4": ("XGBoost", XGBoostPredictor),
            "5": ("Gradient Boosting", GradientBoostingPredictor),
            "6": ("Elastic Net", ElasticNetPredictor),
            "7": ("SVR", SVRPredictor),
            "8": ("KNN", KNNPredictor),
            "9": ("MLP", MLPPredictor),
        }

        model_name, ModelClass = model_map[choice]
        train_single_model(model_name, ModelClass)
    else:
        print_error("Invalid choice! Please choose 0-9")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 3: TEST MODEL
# ============================================================================


def test_all_models():
    """Display test results from in-memory trained models (no .pkl loading)."""
    import time
    from datetime import datetime

    import pandas as pd

    global SESSION_MODELS_DATA, SESSION_TEST_DATA

    # Start total timer
    total_start_time = time.time()

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'DISPLAYING TEST RESULTS'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

    # Get paths
    paths = get_project_paths()
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Display test set information
    if SESSION_TEST_DATA is not None and len(SESSION_TEST_DATA) > 0:
        test_years = SESSION_TEST_DATA.index.get_level_values(1).unique()
        print(f"{Colors.CYAN}Test Set Information:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Test years: {Colors.BOLD}{int(test_years.min())}-{int(test_years.max())}{Colors.RESET} ({len(test_years)} years)"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Test samples: {Colors.BOLD}{len(SESSION_TEST_DATA)}{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} Countries: {Colors.BOLD}{SESSION_TEST_DATA.index.get_level_values(0).nunique()}{Colors.RESET}"
        )
        print()

    # Sort by R^2 score for ranking
    results = [
        {
            "model": m["model"],
            "r2": m["r2"],
            "adj_r2": m.get("adj_r2", m["r2"]),  # Use R² if adj_r2 not available
            "f_stat": m.get("f_stat", 0),
            "f_pvalue": m.get("f_pvalue", 1.0),
            "mae": m["mae"],
            "rmse": m["rmse"],
        }
        for m in SESSION_MODELS_DATA
    ]

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)

    # Professional table display with Adjusted R² and F-statistic
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}╔{'═' * 98}╗{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}║{Colors.RESET} {Colors.BOLD}{'MODEL PERFORMANCE RANKING'.center(96)}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}║{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}╠{'═' * 98}╣{Colors.RESET}")
    # Build header ensuring exactly 96 characters
    header_content = f" {'Rank':^9} │ {'Model':^22} │ {'R²':^8} │ {'Adj R²':^8} │ {'F-stat':^12} │ {'MAE':^8} │ {'RMSE':^8}  "
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}║{Colors.RESET}{Colors.BOLD}{header_content}{Colors.RESET}{Colors.BOLD}{Colors.BRIGHT_WHITE}║{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}╠{'═' * 98}╣{Colors.RESET}")

    # Display models with rank indicators
    for rank, (idx, row) in enumerate(results_df.iterrows(), 1):
        # Rank indicator for top 3
        if rank == 1:
            rank_mark = "***"
        elif rank == 2:
            rank_mark = "** "
        elif rank == 3:
            rank_mark = "*  "
        else:
            rank_mark = "   "

        # Color coding based on R² score
        if row["r2"] >= 0.75:
            score_color = Colors.BRIGHT_GREEN
        elif row["r2"] >= 0.65:
            score_color = Colors.BRIGHT_CYAN
        else:
            score_color = Colors.BRIGHT_WHITE

        # Format F-statistic with significance indicator
        if row["f_pvalue"] < 0.001:
            f_str = f"{row['f_stat']:.1f}***"
        elif row["f_pvalue"] < 0.01:
            f_str = f"{row['f_stat']:.1f}**"
        elif row["f_pvalue"] < 0.05:
            f_str = f"{row['f_stat']:.1f}*"
        else:
            f_str = f"{row['f_stat']:.1f}"

        # Build the content line ensuring exactly 96 characters
        content = (
            f" {rank_mark} {rank:^4} │ {row['model']:<22} │ "
            f"{score_color}{row['r2']:^8.4f}{Colors.RESET} │ "
            f"{row['adj_r2']:^8.4f} │ "
            f"{f_str:^12} │ "
            f"{row['mae']:^8.4f} │ {row['rmse']:^8.4f} "
        )

        # Print with proper width (accounting for color codes, the visible content should be 96 chars)
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_WHITE}║{Colors.RESET}{content}{Colors.BOLD}{Colors.BRIGHT_WHITE}║{Colors.RESET}"
        )

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}╚{'═' * 98}╝{Colors.RESET}")

    # Statistics summary
    print()
    avg_r2 = results_df["r2"].mean()
    std_r2 = results_df["r2"].std()
    avg_adj_r2 = results_df["adj_r2"].mean()
    avg_mae = results_df["mae"].mean()
    best_model = results_df.iloc[0]

    # Calculate padding for proper alignment
    champion_line = f"  Champion:         {best_model['model']:<30} R² = {best_model['r2']:.4f} │ Adj R² = {best_model['adj_r2']:.4f}"
    avg_r2_line = f"  Average R²:       {avg_r2:.4f} ± {std_r2:.4f}"
    avg_adj_r2_line = f"  Average Adj R²:   {avg_adj_r2:.4f}"
    avg_mae_line = f"  Average MAE:      {avg_mae:.4f}"
    models_tested_line = f"  Models tested:    {len(results_df)}/7"

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}┌{'─' * 98}┐{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET} {Colors.BOLD}{'PERFORMANCE STATISTICS'.center(96)}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}├{'─' * 98}┤{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}{champion_line:<96} {Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}"
    )
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}{avg_r2_line:<96} {Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}"
    )
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}{avg_adj_r2_line:<96} {Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}"
    )
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}{avg_mae_line:<96} {Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}"
    )
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}{models_tested_line:<96} {Colors.BOLD}{Colors.BRIGHT_WHITE}│{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}└{'─' * 98}┘{Colors.RESET}")

    # Show total display time
    total_elapsed = time.time() - total_start_time
    print()
    print(f"{Colors.BRIGHT_CYAN}Display time: {total_elapsed:.2f}s{Colors.RESET}")

    # Mark as completed
    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["test_model"] = True

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def test_dynamic_panel_results():
    """Display the most recent Dynamic Panel test results."""
    from pathlib import Path

    import pandas as pd

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'DYNAMIC PANEL TEST RESULTS'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

    paths = get_project_paths()
    results_dir = paths["results"]

    # Check for panel analysis results (fixed filename)
    panel_file = results_dir / "panel_analysis.txt"

    if not panel_file.exists():
        print_warning("No Dynamic Panel results found!")
        print_info("Please run [2] Train Model > [2] Dynamic Panel Analysis first")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    print_info(f"Displaying results from: {panel_file.name}\n")

    # Read and display the results
    try:
        with open(panel_file, "r") as f:
            content = f.read()

        # Extract key metrics from the file
        lines = content.split("\n")
        r2_line = [l for l in lines if "Test R^2:" in l]
        rmse_line = [l for l in lines if "Test RMSE:" in l]

        print(f"{Colors.CYAN}Test Set Performance:{Colors.RESET}\n")
        if r2_line:
            r2_val = r2_line[0].split(":")[1].strip()
            print(
                f"  {Colors.BRIGHT_WHITE}R^2 Score:{Colors.RESET}  {Colors.BOLD}{r2_val}{Colors.RESET}"
            )
        if rmse_line:
            rmse_val = rmse_line[0].split(":")[1].strip()
            print(f"  {Colors.BRIGHT_WHITE}RMSE:{Colors.RESET}      {rmse_val}")

        print()
        print(f"{Colors.CYAN}Full results saved to:{Colors.RESET}")
        print(f"  {Colors.DIM}-> {latest_file}{Colors.RESET}")

    except Exception as e:
        print_error(f"Error reading results: {str(e)}")

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def test_model():
    """Display test results from trained models (in-memory, no .pkl files)."""
    print_section("ACTION 3: TEST MODEL")

    # Check if training was done in this session
    global TRAINING_COMPLETED_THIS_SESSION, SESSION_MODELS_DATA
    if not TRAINING_COMPLETED_THIS_SESSION or SESSION_MODELS_DATA is None:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{'ERROR: TRAINING REQUIRED IN THIS SESSION'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}\n")

        print_error(
            "You must TRAIN models in THIS SESSION before viewing test results!"
        )
        print()
        print(
            f"{Colors.CYAN}Models are stored in memory (not saved as .pkl files).{Colors.RESET}"
        )
        print(
            f"{Colors.CYAN}Test results can only be viewed after training in the current session.{Colors.RESET}"
        )
        print()
        print(f"{Colors.BRIGHT_WHITE}Steps to follow:{Colors.RESET}")
        print(f"  {Colors.BRIGHT_GREEN}1.{Colors.RESET} Return to main menu")
        print(
            f"  {Colors.BRIGHT_GREEN}2.{Colors.RESET} Choose {Colors.BOLD}[2] TRAIN MODEL > [1] ALL MODELS{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}3.{Colors.RESET} After training completes, return here to view test results"
        )
        print()
        print(
            f"{Colors.YELLOW}Important:{Colors.RESET} If you restart the program, you must re-train models."
        )

        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    # Display test results menu
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Test Results Options:{Colors.RESET}")
    print(f"{Colors.DIM}{'' * 80}{Colors.RESET}\n")

    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[A]{Colors.RESET}  View All ML Test Results    {Colors.DIM}->{Colors.RESET}  Display all {len(SESSION_MODELS_DATA)} ML models"
    )
    print()
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[P]{Colors.RESET}  Dynamic Panel Results       {Colors.DIM}->{Colors.RESET}  View panel analysis results"
    )

    print(f"\n{Colors.DIM}{'' * 80}{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_RED}[0]{Colors.RESET}  Back to main menu")

    choice = (
        input(f"\n{Colors.BOLD}Enter choice [0, A, P]: {Colors.RESET}").strip().upper()
    )

    if choice == "0":
        return
    elif choice == "A":
        # Display all ML model test results
        test_all_models()
    elif choice == "P":
        # Show Dynamic Panel results
        test_dynamic_panel_results()
    else:
        print_error("Invalid choice! Please choose 0, A, or P")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 4: EVALUATE SAVED MODELS
# ============================================================================


def evaluate_saved_models():
    """Compare all saved models and show performance metrics."""
    import pandas as pd

    print_section("ACTION 4: EVALUATE SAVED MODELS")

    # Check workflow: Must have tested models in this session
    global WORKFLOW_COMPLETION
    if not WORKFLOW_COMPLETION["test_model"]:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{'WORKFLOW ERROR: TESTING REQUIRED'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}\n")

        print_error("You must TEST models in this session before evaluating!")
        print()
        print(
            f"{Colors.CYAN}Evaluation compares training vs test results from the CURRENT session.{Colors.RESET}"
        )
        print()
        print(f"{Colors.BRIGHT_WHITE}Required workflow:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_GREEN}1.{Colors.RESET} Run {Colors.BOLD}[1] Data Preparation{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}2.{Colors.RESET} Run {Colors.BOLD}[2] Train Model{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}3.{Colors.RESET} Run {Colors.BOLD}[3] Test Model{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}4.{Colors.RESET} Then run {Colors.BOLD}[4] Evaluate Saved Models{Colors.RESET}"
        )
        print()
        print(
            f"{Colors.DIM}Note: If you restart the program, you must re-run all steps.{Colors.RESET}"
        )

        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    paths = get_project_paths()
    results_dir = paths["results"]

    # Check for result files (fixed filenames)
    benchmark_file = results_dir / "benchmark_results.csv"
    test_file = results_dir / "test_results.csv"

    if not benchmark_file.exists() and not test_file.exists():
        print_warning("No results found")
        print_info("Please run:")
        print(
            f"  {Colors.CYAN}[2] Train Model > [1] ALL ML MODELS{Colors.RESET} - to train and benchmark"
        )
        print(
            f"  {Colors.CYAN}[3] Test Model > [A] TEST ALL MODELS{Colors.RESET} - to test on test set"
        )
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Available Results:{Colors.RESET}")
    print(f"{Colors.DIM}{'' * 80}{Colors.RESET}\n")

    # Show training/benchmark results
    if benchmark_file.exists():
        print(f"{Colors.BRIGHT_GREEN}Training Results (Benchmark):{Colors.RESET}")
        print(f"{Colors.DIM}Cross-validation scores during training{Colors.RESET}\n")

        try:
            df = pd.read_csv(benchmark_file)

            # Check if required columns exist
            if "r2" not in df.columns or "model" not in df.columns:
                print(
                    f"  {Colors.BRIGHT_YELLOW}[WARNING] Skipped (missing required columns){Colors.RESET}\n"
                )
            else:
                timestamp = (
                    df["timestamp"].iloc[0] if "timestamp" in df.columns else "N/A"
                )
                print(f"  {Colors.DIM}Date: {timestamp}{Colors.RESET}\n")

                # Show all models
                df_sorted = df.sort_values("r2", ascending=False)
                for idx, row in df_sorted.iterrows():
                    print(
                        f"  {Colors.CYAN}->{Colors.RESET} {row['model']:20s} R^2={row['r2']:.4f}"
                    )
                print()
        except Exception as e:
            print(
                f"  {Colors.BRIGHT_RED}[WARNING] Error reading file: {str(e)}{Colors.RESET}\n"
            )

    # Show test results
    if test_file.exists():
        print(f"{Colors.BRIGHT_GREEN}Test Results:{Colors.RESET}")
        print(f"{Colors.DIM}Performance on held-out test set{Colors.RESET}\n")

        try:
            df = pd.read_csv(test_file)

            # Check if required columns exist
            if "r2" not in df.columns or "model" not in df.columns:
                print(
                    f"  {Colors.BRIGHT_YELLOW}[WARNING] Skipped (missing required columns){Colors.RESET}\n"
                )
            else:
                timestamp = (
                    df["timestamp"].iloc[0] if "timestamp" in df.columns else "N/A"
                )
                print(f"  {Colors.DIM}Date: {timestamp}{Colors.RESET}\n")

                # Show all models
                df_sorted = df.sort_values("r2", ascending=False)
                for idx, row in df_sorted.iterrows():
                    print(
                        f"  {Colors.CYAN}->{Colors.RESET} {row['model']:20s} R^2={row['r2']:.4f}"
                    )
                print()
        except Exception as e:
            print(
                f"  {Colors.BRIGHT_RED}[WARNING] Error reading file: {str(e)}{Colors.RESET}\n"
            )

    # Show latest detailed comparison
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'DETAILED BENCHMARK COMPARISON'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

    if benchmark_file.exists():
        df = pd.read_csv(benchmark_file)
        df_sorted = df.sort_values("r2", ascending=False)

        print(f"{Colors.CYAN}File:{Colors.RESET} {benchmark_file.name}\n")
        print(
            f"{Colors.BOLD}Rank  Model                R^2        MAE       RMSE      Time{Colors.RESET}"
        )
        print(f"{Colors.DIM}{'' * 80}{Colors.RESET}")

        for idx, row in df_sorted.iterrows():
            rank = idx + 1
            print(
                f"{rank:4d}  {row['model']:18s}  {Colors.BOLD}{row['r2']:.4f}{Colors.RESET}    {row['mae']:.4f}    {row['rmse']:.4f}    {row['time']:.1f}s"
            )

        best = df_sorted.iloc[0]
        print()
        print(
            f"{Colors.BRIGHT_GREEN} Best: {best['model']} (R^2 = {best['r2']:.4f}){Colors.RESET}"
        )

    # Create comprehensive evaluation CSV
    print()
    print(f"{Colors.CYAN}Creating comprehensive evaluation CSV...{Colors.RESET}")

    all_results = []

    # Add benchmark results
    if benchmark_file.exists():
        df = pd.read_csv(benchmark_file)
        df["source"] = "training"
        df["file"] = benchmark_file.stem
        all_results.append(df)

    # Add test results
    if test_file.exists():
        df = pd.read_csv(test_file)
        df["source"] = "test"
        df["file"] = test_file.stem
        all_results.append(df)

    if all_results:
        # Combine all dataframes
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save comprehensive CSV
        eval_file = results_dir / "evaluation_summary.csv"
        combined_df.to_csv(eval_file, index=False)

        print_success(f"Evaluation summary saved: {eval_file.name}")
        print()
        print(f"{Colors.CYAN}Output file:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_WHITE}-{Colors.RESET} {eval_file.name} ({eval_file.stat().st_size / 1024:.1f} KB)"
        )
        print(f"  {Colors.DIM}-> Contains all training and test results{Colors.RESET}")
        print(
            f"  {Colors.DIM}-> {len(combined_df)} total entries from training and test runs{Colors.RESET}"
        )

    # Mark as completed
    WORKFLOW_COMPLETION["evaluate_models"] = True

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 5: RUN VISUALIZATION
# ============================================================================
# Note: All visualization logic is in src/evaluation.py
# Main.py only coordinates and calls these functions with proper paths

from src.evaluation import (
    generate_cv_scores_comparison,
    generate_error_distribution,
    generate_feature_importance_plot,
    generate_learning_curves,
    generate_model_comparison_chart,
    generate_political_stability_evolution,
    generate_predictions_plot,
    generate_regional_analysis,
    generate_residual_plots,
    generate_statistical_analysis,
    generate_time_series_predictions,
)


def call_viz_function(func, func_name):
    """
    Wrapper to call visualization functions from src.evaluation.

    Uses in-memory models from SESSION_MODELS_DATA (no .pkl loading).
    """
    print()
    print(f"{Colors.CYAN}-> Generating {func_name}...{Colors.RESET}")

    global SESSION_MODELS_DATA, SESSION_TRAIN_DATA, SESSION_TEST_DATA

    paths = get_project_paths()
    figures_dir = paths["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Pass in-memory models and data (no .pkl loading!)
        if func.__name__ in [
            "generate_statistical_analysis",
            "generate_political_stability_evolution",
        ]:
            output_file = func(SESSION_TEST_DATA, figures_dir)
        elif func.__name__ in [
            "generate_learning_curves",
            "generate_cv_scores_comparison",
        ]:
            output_file = func(SESSION_MODELS_DATA, SESSION_TRAIN_DATA, figures_dir)
        elif func.__name__ in [
            "generate_time_series_predictions",
            "generate_regional_analysis",
        ]:
            output_file = func(SESSION_MODELS_DATA, SESSION_TEST_DATA, figures_dir)
        else:
            # generate_model_comparison_chart, generate_feature_importance_plot,
            # generate_predictions_plot, generate_residual_plots, generate_error_distribution
            output_file = func(SESSION_MODELS_DATA, figures_dir)

        if output_file:
            print_success(f"Saved: {output_file.name}")
            print(f"  {Colors.DIM}Location: {output_file}{Colors.RESET}")
        return output_file
    except Exception as e:
        print_error(f"Failed to generate {func_name}: {str(e)}")
        return None


def run_visualization():
    """Generate plots and charts using in-memory trained models."""
    print_section("ACTION 5: RUN VISUALIZATION")

    # Check if models are available in session (NO .pkl files!)
    global SESSION_MODELS_DATA, SESSION_TRAIN_DATA, SESSION_TEST_DATA
    if SESSION_MODELS_DATA is None:
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_RED}{'WORKFLOW ERROR: TRAINING REQUIRED IN THIS SESSION'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{'!' * 80}{Colors.RESET}\n")

        print_error(
            "You must TRAIN models in THIS SESSION before generating visualizations!"
        )
        print()
        print(
            f"{Colors.CYAN}Models are stored in memory (not saved to disk).{Colors.RESET}"
        )
        print(
            f"{Colors.CYAN}Visualizations can only be generated after training in the current session.{Colors.RESET}"
        )
        print()
        print(f"{Colors.BRIGHT_WHITE}Required workflow:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_GREEN}1.{Colors.RESET} Run {Colors.BOLD}[1] Data Preparation{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}2.{Colors.RESET} Run {Colors.BOLD}[2] Train Model > [1] ALL MODELS{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_GREEN}3.{Colors.RESET} Then run {Colors.BOLD}[5] Visualization{Colors.RESET}"
        )
        print()
        print(
            f"{Colors.YELLOW}Important:{Colors.RESET} If you restart the program, you must re-train models."
        )
        print(
            f"{Colors.DIM}(Models are NOT saved to .pkl files - they exist only in memory){Colors.RESET}"
        )

        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Choose visualization:{Colors.RESET}")
    print(f"{Colors.DIM}{'' * 80}{Colors.RESET}\n")

    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[1]{Colors.RESET}   Model Comparison Chart      {Colors.DIM}->{Colors.RESET}  Compare R^2, MAE, RMSE"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[2]{Colors.RESET}   Feature Importance          {Colors.DIM}->{Colors.RESET}  Most important features"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[3]{Colors.RESET}   Predictions vs Actual       {Colors.DIM}->{Colors.RESET}  Scatter plots"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[4]{Colors.RESET}   Statistical Analysis        {Colors.DIM}->{Colors.RESET}  Distribution & correlation"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[5]{Colors.RESET}   Residual Plots              {Colors.DIM}->{Colors.RESET}  Model diagnostic plots"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[6]{Colors.RESET}   Learning Curves             {Colors.DIM}->{Colors.RESET}  Overfitting/underfitting {Colors.YELLOW}(slow){Colors.RESET}"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[7]{Colors.RESET}   Time Series Predictions     {Colors.DIM}->{Colors.RESET}  Temporal evolution"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[8]{Colors.RESET}   Cross-Validation Scores     {Colors.DIM}->{Colors.RESET}  Model robustness"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[9]{Colors.RESET}   Error Distribution          {Colors.DIM}->{Colors.RESET}  Prediction errors analysis"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[10]{Colors.RESET}  Regional Analysis           {Colors.DIM}->{Colors.RESET}  Geographic performance"
    )
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}[11]{Colors.RESET}  Political Stability Evolution {Colors.DIM}->{Colors.RESET} Global average over time"
    )
    print()
    print(
        f"  {Colors.BOLD}{Colors.BRIGHT_GREEN}[12]{Colors.RESET}  ALL VISUALIZATIONS          {Colors.DIM}->{Colors.RESET}  Generate 10 plots (fast)"
    )

    print(f"\n{Colors.DIM}{'' * 80}{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_RED}[0]{Colors.RESET}   Back to main menu")

    choice = input(f"\n{Colors.BOLD}Enter choice [0-12]: {Colors.RESET}").strip()

    if choice == "0":
        return

    if choice == "12":
        # Generate all visualizations
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'GENERATING ALL VISUALIZATIONS'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

        call_viz_function(generate_model_comparison_chart, "Model Comparison Chart")
        call_viz_function(generate_feature_importance_plot, "Feature Importance Plot")
        call_viz_function(generate_predictions_plot, "Predictions vs Actual")
        call_viz_function(generate_statistical_analysis, "Statistical Analysis")
        call_viz_function(generate_residual_plots, "Residual Plots")
        # Learning Curves removed - too slow (available individually as option 6)
        # call_viz_function(generate_learning_curves, "Learning Curves")
        call_viz_function(generate_time_series_predictions, "Time Series Predictions")
        call_viz_function(generate_cv_scores_comparison, "Cross-Validation Scores")
        call_viz_function(generate_error_distribution, "Error Distribution")
        call_viz_function(generate_regional_analysis, "Regional Analysis")
        call_viz_function(
            generate_political_stability_evolution, "Political Stability Evolution"
        )

        print()
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_GREEN}OK All 10 visualizations generated successfully!{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}(Learning Curves skipped - available individually as option 6){Colors.RESET}"
        )

        # Mark as completed
        WORKFLOW_COMPLETION["visualization"] = True

    elif choice == "1":
        call_viz_function(generate_model_comparison_chart, "Model Comparison Chart")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "2":
        call_viz_function(generate_feature_importance_plot, "Feature Importance Plot")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "3":
        call_viz_function(generate_predictions_plot, "Predictions vs Actual")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "4":
        call_viz_function(generate_statistical_analysis, "Statistical Analysis")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "5":
        call_viz_function(generate_residual_plots, "Residual Plots")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "6":
        call_viz_function(generate_learning_curves, "Learning Curves")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "7":
        call_viz_function(generate_time_series_predictions, "Time Series Predictions")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "8":
        call_viz_function(generate_cv_scores_comparison, "Cross-Validation Scores")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "9":
        call_viz_function(generate_error_distribution, "Error Distribution")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "10":
        call_viz_function(generate_regional_analysis, "Regional Analysis")
        WORKFLOW_COMPLETION["visualization"] = True
    elif choice == "11":
        call_viz_function(
            generate_political_stability_evolution, "Political Stability Evolution"
        )
        WORKFLOW_COMPLETION["visualization"] = True
    else:
        print_error("Invalid choice! Please choose 0-12")

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 6: SHOW DASHBOARD LINK
# ============================================================================


def show_dashboard_link():
    """Display Streamlit dashboard URL and open in browser."""
    import webbrowser

    print_section("ACTION 6: SHOW DASHBOARD LINK")

    dashboard_url = "http://localhost:8501"

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'STREAMLIT DASHBOARD'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

    # Check if dashboard is already running
    is_running = False
    try:
        import requests

        response = requests.get(dashboard_url, timeout=1)
        if response.status_code == 200:
            is_running = True
            print_success(f"Dashboard is running!")
    except:
        print_warning("Dashboard is not currently running")

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Dashboard URL:{Colors.RESET}")
    print(f"  {Colors.BRIGHT_CYAN}{Colors.UNDERLINE}{dashboard_url}{Colors.RESET}")
    print()

    if not is_running:
        print(f"{Colors.YELLOW}To start the dashboard:{Colors.RESET}")
        print(f"  {Colors.DIM}streamlit run main/dashboard.py{Colors.RESET}")
        print()

    # Offer to open in browser
    print(f"{Colors.CYAN}Options:{Colors.RESET}")
    print(f"  {Colors.CYAN}[1]{Colors.RESET}  Open dashboard in browser")
    if not is_running:
        print(f"  {Colors.CYAN}[2]{Colors.RESET}  Launch dashboard and open")
    print(f"  {Colors.BRIGHT_RED}[0]{Colors.RESET}  Back to menu")

    choice = input(f"\n{Colors.BOLD}Enter choice: {Colors.RESET}").strip()

    if choice == "1":
        if is_running:
            print()
            print_info("Opening dashboard in browser...")
            webbrowser.open(dashboard_url)
            print_success(f"Browser opened: {dashboard_url}")
        else:
            print()
            print_error("Dashboard is not running. Please start it first.")
    elif choice == "2" and not is_running:
        print()
        print_info("Launching dashboard...")
        try:
            import subprocess

            # Kill any existing streamlit processes first
            try:
                subprocess.run(
                    ["pkill", "-f", "streamlit"], capture_output=True, timeout=5
                )
                import time

                time.sleep(1)  # Wait for processes to die
            except:
                pass  # Ignore if pkill fails (Windows or no processes)

            dashboard_path = PROJECT_ROOT / "main" / "dashboard.py"

            # Verify file exists
            if not dashboard_path.exists():
                print_error(f"Dashboard file not found at: {dashboard_path}")
                print_info(f"Looking for file...")
                print_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
                print_info(f"dashboard_path = {dashboard_path}")
            else:
                print_success(f"Dashboard file found: {dashboard_path}")

            # Launch streamlit (show output for debugging)
            process = subprocess.Popen(
                ["streamlit", "run", str(dashboard_path)],
                cwd=str(PROJECT_ROOT),  # Set working directory to project root
            )
            print_success("Dashboard launched!")
            print_info("Waiting for dashboard to start...")

            # Wait a bit for dashboard to start
            import time

            time.sleep(3)

            print_info("Opening in browser...")
            webbrowser.open(dashboard_url)
            print_success(f"Browser opened: {dashboard_url}")
        except Exception as e:
            print_error(f"Failed to launch dashboard: {str(e)}")

    # Mark as completed (user viewed dashboard info)
    global WORKFLOW_COMPLETION
    WORKFLOW_COMPLETION["dashboard"] = True

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# ACTION 7: TEST COVERAGE
# ============================================================================


def test_coverage():
    """Run test coverage analysis and display results."""
    print_section("ACTION 7: TEST COVERAGE")

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'RUNNING TEST COVERAGE ANALYSIS'.center(80)}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

    paths = get_project_paths()
    project_root = paths["root"]
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print_error("Tests directory not found!")
        print_info(f"Expected location: {tests_dir}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return

    print_info("Step 1/3: Running tests with coverage tracking...")
    print(f"{Colors.DIM}This may take a minute...{Colors.RESET}\n")

    try:
        # Clean old coverage data
        import subprocess

        coverage_file = project_root / ".coverage"
        if coverage_file.exists():
            coverage_file.unlink()

        # Run tests with coverage using pytest --cov (more reliable than coverage run)
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/", "--cov=src", "--cov-report=", "-v"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=180,
        )

        # Count test results
        lines = result.stdout.split("\n")
        summary_line = [l for l in lines if "passed" in l or "skipped" in l]

        if summary_line:
            print_success("Tests completed!")
            print(f"  {Colors.DIM}{summary_line[-1].strip()}{Colors.RESET}")
        else:
            print_success("Tests completed")

        print()
        print_info("Step 2/3: Generating coverage report...")

        # Generate coverage report
        result = subprocess.run(
            ["python3", "-m", "coverage", "report", "--include=src/*"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print()
            print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}")
            print(
                f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'COVERAGE REPORT'.center(80)}{Colors.RESET}"
            )
            print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'=' * 80}{Colors.RESET}\n")

            # Display coverage report
            print(f"{Colors.CYAN}{result.stdout}{Colors.RESET}")

            # Parse total coverage
            lines = result.stdout.split("\n")
            total_line = [l for l in lines if "TOTAL" in l]

            if total_line:
                parts = total_line[0].split()
                if len(parts) >= 4:
                    coverage_pct = parts[-1].rstrip("%")
                    try:
                        coverage_val = float(coverage_pct)

                        print()
                        if coverage_val >= 70:
                            print(
                                f"{Colors.BRIGHT_GREEN}OK Coverage is {coverage_pct}% (Target: 70-75%){Colors.RESET}"
                            )
                        elif coverage_val >= 60:
                            print(
                                f"{Colors.BRIGHT_YELLOW}[WARNING] Coverage is {coverage_pct}% (Close to target: 70-75%){Colors.RESET}"
                            )
                        else:
                            print(
                                f"{Colors.BRIGHT_RED}[X] Coverage is {coverage_pct}% (Below target: 70-75%){Colors.RESET}"
                            )
                    except ValueError:
                        pass

        print()
        print_info("Step 3/3: Generating HTML coverage report...")

        # Generate HTML report
        html_result = subprocess.run(
            ["python3", "-m", "coverage", "html", "--include=src/*", "-d", "htmlcov"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if html_result.returncode == 0:
            html_path = project_root / "htmlcov" / "index.html"
            print_success(f"HTML report generated: {html_path}")
            print()
            print(f"{Colors.CYAN}To view detailed HTML report:{Colors.RESET}")
            print(f"  {Colors.DIM}open {html_path}{Colors.RESET}")
        else:
            print_warning("HTML report generation failed")
            if html_result.stderr:
                print()
                print(f"{Colors.DIM}Error details:{Colors.RESET}")
                print(html_result.stderr)
            # Try alternate command with pytest directly
            print()
            print_info("Attempting alternative method with pytest...")
            alt_result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "pytest",
                    "--cov=src",
                    "--cov-report=html",
                    "--co",
                    "-q",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if alt_result.returncode == 0:
                html_path = project_root / "htmlcov" / "index.html"
                if html_path.exists():
                    print_success(f"HTML report generated via pytest: {html_path}")
                    print()
                    print(f"{Colors.CYAN}To view detailed HTML report:{Colors.RESET}")
                    print(f"  {Colors.DIM}open {html_path}{Colors.RESET}")
                else:
                    print_warning("HTML directory not found after pytest attempt")

        # Display final coverage summary in terminal
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'FINAL COVERAGE SUMMARY'.center(80)}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

        final_report = subprocess.run(
            ["python3", "-m", "coverage", "report", "--include=src/*"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if final_report.returncode == 0:
            print(f"{Colors.CYAN}{final_report.stdout}{Colors.RESET}")
        else:
            print_warning("Could not generate final coverage summary")

        # Mark as completed
        global WORKFLOW_COMPLETION
        WORKFLOW_COMPLETION["test_coverage"] = True

    except subprocess.TimeoutExpired:
        print()
        print_error("Test execution timed out (>3 minutes)")
        print_info("Some tests may be taking too long. Please check test.py")

    except FileNotFoundError:
        print()
        print_error("Coverage tool not found!")
        print()
        print(f"{Colors.CYAN}Please install coverage:{Colors.RESET}")
        print(f"  {Colors.DIM}pip install coverage pytest{Colors.RESET}")

    except Exception as e:
        print()
        print_error(f"An error occurred: {str(e)}")

    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# MAIN MENU
# ============================================================================


def show_main_menu():
    """Display the main menu."""
    global WORKFLOW_COMPLETION

    clear_screen()
    print_header("POLITICAL STABILITY PREDICTION", "ML & Econometric Analysis Pipeline")

    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}PROJECT WORKFLOW{Colors.RESET}")
    print(f"{Colors.DIM}{'' * 80}{Colors.RESET}\n")

    # Helper function to get menu item color based on completion status
    def menu_item(number, key, title, description):
        completed = WORKFLOW_COMPLETION.get(key, False)
        color = Colors.BRIGHT_GREEN if completed else Colors.BRIGHT_CYAN
        return f"  {Colors.BOLD}{color}[{number}]{Colors.RESET}  {title:24s}{Colors.DIM}->{Colors.RESET}  {description}"

    print(
        menu_item(
            "0",
            "check_environment",
            "Check Environment",
            "Verify dependencies and data",
        )
    )
    print(
        menu_item(
            "1", "data_preparation", "Run Data Preparation", "Load and prepare datasets"
        )
    )
    print(menu_item("2", "train_model", "Train Model", "Train ML/econometric models"))
    print(menu_item("3", "test_model", "Test Model", "Test trained models"))
    print(
        menu_item("4", "evaluate_models", "Evaluate Saved Models", "Compare all models")
    )
    print(
        menu_item(
            "5", "visualization", "Run Visualization", "Generate plots and charts"
        )
    )
    print(
        menu_item(
            "6", "dashboard", "Show Dashboard Link", "Display Streamlit dashboard"
        )
    )
    print(menu_item("7", "test_coverage", "Test Coverage", "Run coverage analysis"))

    print(f"\n{Colors.DIM}{'' * 80}{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.BRIGHT_RED}[Q]{Colors.RESET}  Quit")
    print()


def main():
    """Main entry point."""
    # Get project paths
    paths = get_project_paths()

    # Main loop
    while True:
        show_main_menu()

        choice = (
            input(f"{Colors.BOLD}Enter choice [0-7, Q]: {Colors.RESET}").strip().upper()
        )

        if choice == "Q":
            print()
            print_info("Exiting program...")
            break

        elif choice == "0":
            check_environment()

        elif choice == "1":
            run_data_preparation()

        elif choice == "2":
            train_model()

        elif choice == "3":
            test_model()

        elif choice == "4":
            evaluate_saved_models()

        elif choice == "5":
            run_visualization()

        elif choice == "6":
            show_dashboard_link()

        elif choice == "7":
            test_coverage()

        else:
            print_error("Invalid choice! Please choose numbers 0-7 or Q to quit")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")

    print()
    print(
        f"{Colors.BRIGHT_CYAN}Thank you for using Political Stability Prediction!{Colors.RESET}"
    )
    print()


if __name__ == "__main__":
    main()
