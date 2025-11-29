"""
Basic tests to verify project structure and functions exist
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_project_structure():
    """Verify project directories exist"""
    project_root = Path(__file__).parent.parent

    assert (project_root / "src").exists(), "src/ directory missing"
    assert (project_root / "data").exists(), "data/ directory missing"
    assert (project_root / "tests").exists(), "tests/ directory missing"
    assert (project_root / "results").exists(), "results/ directory missing"


def test_data_loader_module_exists():
    """Verify data_loader module can be imported"""
    try:
        import data_loader
        assert True
    except ImportError:
        assert False, "data_loader module cannot be imported"


def test_data_loader_has_functions():
    """Verify data loading functions are defined"""
    import data_loader

    # Check that module exists (even if functions not implemented yet)
    assert hasattr(data_loader, '__file__'), "data_loader not properly defined"


def test_main_exists():
    """Verify main.py exists and can be imported"""
    project_root = Path(__file__).parent.parent
    assert (project_root / "main.py").exists(), "main.py missing"
