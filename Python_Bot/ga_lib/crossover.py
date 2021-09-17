from typing import Callable
import numpy as np

CrossoverFunction_Type = Callable[[np.ndarray, np.ndarray], np.ndarray]

# idx_parent_A = np.nonzero(self.probability_choose>=np.random.rand(1))[0]
# idx_parent_B = np.nonzero(self.probability_choose>=np.random.rand(1))[0]

def Crossover_Each_Gene(parent_A: np.ndarray, parent_B: np.ndarray) -> np.ndarray:
    gene_index = np.random.rand(1,len(parent_A))>=0.5
    return gene_index*parent_A + (1-gene_index)* parent_B