"""
Logging Configuration
Centralized logging setup for panel data analysis
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging with file and console output.

    Parameters
    ----------
    log_dir : Path, optional
        Directory for log files. Defaults to 'logs' in current directory
    level : int, default=logging.INFO
        Logging level

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"

    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('political_stability')
    logger.info(f"Logging initialized: {log_file}")
    return logger
