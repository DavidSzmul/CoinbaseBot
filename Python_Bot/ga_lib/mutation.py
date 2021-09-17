from typing import Callable
import numpy as np

MutationFunction_Type = Callable[[np.ndarray, float], np.ndarray]