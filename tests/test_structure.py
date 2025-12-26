"""
Project structure tests.
Tests for verifying project directory structure and module imports.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_project_structure():
    # Test: Verifies that all project directories exist
    # Purpose: Ensures structure integrity before execution
    """Verify project directories exist."""
    project_root = Path(__file__).parent.parent
    assert (project_root / "src").exists()
    assert (project_root / "data").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "main").exists()


def test_module_imports():
    # Test: Verifies that all main modules import without error
    # Purpose: Detects dependency issues or circular imports
    """Test that all main modules can be imported."""
    from src import data_loader, evaluation, models

    assert data_loader is not None
    assert models is not None
    assert evaluation is not None
