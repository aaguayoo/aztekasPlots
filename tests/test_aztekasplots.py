"""Package related tests."""
from aztekasPlot import __version__


def test_version():
    """Checks correct package version."""
    assert __version__ == "0.3.2"
