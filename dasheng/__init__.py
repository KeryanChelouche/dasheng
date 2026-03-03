import importlib.metadata
try:
    __version__ = importlib.metadata.version("dasheng")
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"

from .pretrained.pretrained import (
    dasheng_base,
    dasheng_06B,
    dasheng_12B,
    dasheng_base_spectrogram,
    dasheng_06B_spectrogram,
    dasheng_12B_spectrogram,
)
