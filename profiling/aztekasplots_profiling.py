"""Profiling file."""
from memory_profiler import profile
from profiling.confprofiling import get_source


@get_source(source_path="", source_name="")
@profile
def profiling_aztekasplot_model(source=None):
    """Profiling for model."""
    pass


if __name__ == "__main__":

    profiling_aztekasplot_model()
