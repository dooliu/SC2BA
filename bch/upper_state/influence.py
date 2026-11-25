"""The signed influence matrix and the original RGB histogram hash."""

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from PIL import Image
from scipy.signal import convolve2d

GRID_SIZE = 128
KINDS = ("im-hash", "im-medoid", "im-grid", "grid-zero", "grid-density", "unit-set")
NEURAL_KINDS = KINDS[2:]


def influence_window(observation):
    # PySC2's original array stores x first and displays its transpose.
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    for sign, side in ((1, observation.friendly), (-1, observation.enemy)):
        for unit in side:
            x = min(GRID_SIZE - 1, max(0, int(unit.x)))
            y = min(GRID_SIZE - 1, max(0, int(observation.map_height - unit.y)))
            grid[y, x] += sign * 16
            for radius, weight in enumerate((9, 4, 1), 1):
                grid[max(0, y - radius):y + radius + 1,
                     max(0, x - radius):x + radius + 1] += sign * weight
                grid[y, x] -= sign * weight
    rows, cols = grid.nonzero()
    if not len(rows):
        return grid[40:44, 40:44]
    return grid[max(0, rows.min() - 2):rows.max() + 3, max(0, cols.min() - 2):cols.max() + 3]


class HistogramHash:
    """Keep the legacy RdBu rendering and per-channel equalization."""

    def __init__(self):
        from seaborn import heatmap

        self.heatmap = heatmap
        figure = Figure(figsize=(6.4, 4.8), dpi=100)
        self.canvas = FigureCanvasAgg(figure)
        self.axes = figure.add_subplot()

    def __call__(self, window, scale):
        self.heatmap(window, ax=self.axes, cmap="RdBu", square=True, cbar=False,
                     xticklabels=False, yticklabels=False,
                     norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale))
        self.canvas.draw()
        image = Image.frombytes("RGB", self.canvas.get_width_height(), self.canvas.buffer_rgba())
        self.axes.clear()
        resized = np.asarray(image.resize((256, 256)).convert("RGB"))
        result = []
        for channel in cv2.split(resized):
            counts = cv2.calcHist([cv2.equalizeHist(channel)], [0], None, [8], [0, 256]).ravel()
            result.extend(format(int(count / counts.max() * 15.9), "X") for count in counts)
        return "".join(result)


def grid_input(kind, observation):
    window = influence_window(observation)
    height, width = window.shape
    row, col = (GRID_SIZE - height) // 2, (GRID_SIZE - width) // 2
    image = np.zeros((1 if kind == "im-grid" else 2, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    signed = window / observation.influence_scale
    image[0, row:row + height, col:col + width] = signed
    if kind == "grid-density":
        offsets = np.arange(-5, 6)
        disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= 25
        coverage = convolve2d(np.maximum(signed, 0), disk, mode="same")
        image[1, row:row + height, col:col + width] = coverage * np.maximum(-signed, 0)
    return image
