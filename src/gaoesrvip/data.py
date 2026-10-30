"""Selected ThAr cutouts; coordinates and sampling widths are in analysis-grid pixels."""
from copy import deepcopy
from dataclasses import dataclass, field, replace
import numpy as np

DATA_FIELDS = ('order', 'x', 'y', 'noise', 'mask', 'x0', 'dv', 'line_ids',
               'fit_wave', 'lab_wave', 'table_pixel')

@dataclass
class OrderData:
    order: int
    x: np.ndarray
    y: np.ndarray
    noise: np.ndarray
    mask: np.ndarray
    x0: np.ndarray
    dv: np.ndarray
    line_ids: np.ndarray
    fit_wave: np.ndarray | None = None
    lab_wave: np.ndarray | None = None
    table_pixel: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.order = int(self.order)
        self.metadata = deepcopy(self.metadata)
        for name in DATA_FIELDS[1:]:
            value = getattr(self, name)
            setattr(self, name, np.asarray(value) if value is not None else np.full(len(self.x0), np.nan))

    def select(self, indices):
        indices = np.atleast_1d(indices)
        return replace(self, **{name: getattr(self, name)[indices] for name in DATA_FIELDS[1:]})
