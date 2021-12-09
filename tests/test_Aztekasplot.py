"""Package related tests."""
from Aztekasplot import __version__


def test_version():
    """Checks correct package version."""
    assert __version__ == "0.2.0"
