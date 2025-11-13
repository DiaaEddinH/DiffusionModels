from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("diffusion-models")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.1"

__all__ = ["__version__"]
