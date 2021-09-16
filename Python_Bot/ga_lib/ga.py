from typing import Callable, Tuple
import numpy as np
import numpy.matlib as mb
from dataclasses import dataclass
    
@dataclass
class GA_Params:
    '''Datalass containing all parameters useful to Genetic Algoritm class'''
    range_min: np.ndarray
    range_max: np.ndarray
    nb_population: int
    nb_generation_max: int
    mutation_factor: float=0.1
    keep_best: bool=True
    type_optim: str='min' # 'min' or 'max'
    nb_no_improve_max: int=None


@dataclass    
class Population:
    '''Population used for GA algorithm'''
    X: np.ndarray           # Chromosomes of population
    cost: np.ndarray        # Result of optimization function
    p_selection: np.ndarray # Probaility of selection
    best: np.ndarray        # Chromosomes of current best    
    probability_choose: np.ndarray  # Cumsum of proba to choose parents

    def __init__(self, nb_population: int, nb_params: int, keep_best: bool=True):
        self.X = np.zeros((nb_population, nb_params))
        self._reinit()
        self.keep_best = keep_best

    def _reinit(self):
        nb_population, nb_params = self.X.shape
        self.cost = np.zeros((nb_population,))
        self.p_selection = np.zeros((nb_population,))
        self.best = np.zeros((nb_params,))

    def set_X(self, X: np.ndarray):
        self.X = X

    def update_costs(self, optim_fcn: Callable[[np.ndarray],float], is_minimization: int):
        '''Update costs of population'''
        # Cost
        for idx_pop in len(self.cost):
            self.cost[idx_pop] = optim_fcn(self.X[idx_pop,:])
            if is_minimization<0: # In case of maximization
                self.cost[idx_pop] = 1/self.cost[idx_pop]
        
        # Probability (<1)
        self.p_selection = (1/self.cost)/np.sum(1/self.cost)

    def do_selection(self):
        '''Sort members of population based on costs'''

        # Reorganize from best to worst
        index_sort = np.argsort(self.cost)
        self.X = self.X[index_sort,:];                                    
        self.cost = self.cost[index_sort];  
        self.p_selection = self.p_selection[index_sort]                    
        self.best = self.X[0,:]; 

        self.probability_choose = np.cumsum(self.p_selection)

    def choose_2_parents(self) -> Tuple[np.nd_array, np.nd_array]:

        idx_parent_A = np.nonzero(self.probability_choose>=np.random.rand(1))[0]
        idx_parent_B = np.nonzero(self.probability_choose>=np.random.rand(1))[0]
        return self.X[idx_parent_A, :], self.X[idx_parent_B, :]

CrossoverFunction_Type = Callable[[np.ndarray, np.ndarray], np.ndarray]
MutationFunction_Type = Callable[[np.ndarray, float], np.ndarray]

class GA:
    '''Genetic algorithm class'''

    params:GA_Params
    is_minimization: int
    nb_params: int

    def __init__(self, params: GA_Params=GA_Params()):
        self.set_params(params)

    def set_params(self, params: GA_Params):
        if params.type_optim not in ['min','max']:
            raise ValueError('type_optim attribute not well defined')
        self.is_minimization = 1 if (params.type_optim == 'min') else -1
        self.nb_params = len(self.params.range_min)
        self.params = params


    def _get_random_population_X(self):
        '''Generate a complete random set of chrommosomes'''
        nb_params = self.nb_params
        nb_population = self.params.nb_population

        return (mb.repmat(self.params.range_min, nb_population, 1) + 
                    np.random.rand(nb_population, nb_params)*mb.repmat(self.params.range_max-self.params.range_min, nb_population, 1)
                )   

    def do_crossover(self, population: Population, crossover_fcn: CrossoverFunction_Type) -> np.ndarray:
        '''Creation of childs based on current population'''
        # Choose 2 parents
        childs = population.X.copy() # Keep best on index 0
        for idx_pop in range(self.params.keep_best*1, childs.shape[0]):
            parent_A, parent_B = population.choose_2_parents()
            childs[idx_pop, :] = crossover_fcn(parent_A, parent_B)

        return childs

    def do_mutation(self, childs: np.ndarray, mutation_fcn: MutationFunction_Type) -> np.ndarray:
        '''Mutation of childs'''
        childs_mutated = childs.copy()  # Keep best on index 0
        for idx_pop in range(self.params.keep_best*1, childs_mutated.shape[0]):
            childs_mutated[idx_pop, :] = mutation_fcn(childs_mutated[idx_pop, :], self.params.mutation_factor)
        return childs_mutated

    def run(self,   optim_fcn: Callable[[np.ndarray],float], 
                    crossover_fcn: CrossoverFunction_Type,
                    mutation_fcn: MutationFunction_Type,
                    callback_loop: Callable=None) -> np.ndarray:
        '''Execution of ga using optimization function'''

        # Initialization of internal variables
        ctr_no_improve = 0
        best_cost = None

        # Initialization of population
        current_population = Population(self.params.nb_population, self.nb_params)
        current_population.X = self._get_random_population_X()

        # Loop each generation
        for idx_gen in range(self.params.nb_generation_max):

            ### Costs of population
            current_population.update_costs(optim_fcn, self.is_minimization)
            ### Selection
            current_population.do_selection()

            
            ### Estimation of performances
            current_best_cost = current_population.cost[0]
            if (current_best_cost < best_cost):
                best_cost = current_best_cost
                ctr_no_improve = 0
            else:
                ctr_no_improve += 1

            if callback_loop:
                callback_loop(current_population)
            print(f'Generation {idx_gen} -> best: {current_best_cost} / nb_no_improve: {ctr_no_improve} / mean: {np.mean(current_population.cost)}')
            # End loop if no improvement
            if ctr_no_improve>=self.params.nb_no_improve_max:
                break

            # Prepare new generation
            ### Crossover
            childs = self.do_crossover(current_population, crossover_fcn)
            ### Mutation
            childs_mutated = self.do_mutation(childs, mutation_fcn)
            current_population.set_X(childs_mutated)
        
        return current_population.best




            
            
            
