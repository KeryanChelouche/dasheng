from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch


class SpectrogramDataset(ABC):
    """Abstract base class for all spectrogram datasets.

    Concrete subclasses are responsible for:
      - Locating data files on disk.
      - Defining the cross-validation strategy.
      - Loading and preprocessing a single sample into a tensor
        that is ready to be fed into a FeatureExtractor.

    All datasets produce spectrogram tensors of shape (1, F, T).
    For audio datasets, the conversion to spectrogram is done inside
    load_item() so the rest of the pipeline stays format-agnostic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this dataset (used as cache key)."""
        ...

    @property
    @abstractmethod
    def n_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        ...

    @abstractmethod
    def items(self) -> Tuple[List[Path], np.ndarray]:
        """Return all dataset items and their integer class labels.

        Returns:
            items:  List of file paths (one per sample).
            labels: Integer class label array, shape (N,).
        """
        ...

    @abstractmethod
    def load_item(self, item: Path) -> torch.Tensor:
        """Load and preprocess one sample into a model-ready tensor.

        Returns:
            Spectrogram tensor of shape (1, F, T), float32.
        """
        ...

    @abstractmethod
    def cv_splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each CV fold."""
        ...
