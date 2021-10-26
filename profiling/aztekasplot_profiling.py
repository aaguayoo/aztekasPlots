"""Profiling file.

Here you can code your paths for Gauss model and check  how much time it takes.
"""
from memory_profiler import profile
from profiling.confprofiling import get_source

from Aztekasplot.model import Model


@get_source(source_path="", source_name="")
@profile
def profiling_aztekasplot_model(source=None):
    """Profiling for model."""
    print(Model)
    pass


if __name__ == "__main__":

    profiling_aztekasplot_model()
