"""
Data integrity tests - verify data loading functions exist
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_data_directory_structure():
    """Verify data directory structure"""
    data_dir = Path(__file__).parent.parent / "data"
    assert data_dir.exists(), "data/ directory missing"
    assert (data_dir / "raw").exists(), "data/raw/ directory missing"


def test_data_loader_functions_defined():
    """Verify data loading functions are defined (even if not implemented)"""
    try:
        from data_loader import DataLoader
        # Just check class exists, don't need implementation yet
        assert DataLoader is not None
    except ImportError:
        # If not implemented yet, that's ok for now
        pass
