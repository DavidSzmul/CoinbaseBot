from typing import Callable
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
    mutation_factor: float
    type_optim: str='min' # 'min' or 'max'
    nb_no_improve_max: int=None

CrossoverFunction_Type = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass    
class Population:
    '''Population used for GA algorithm'''
    X: np.ndarray           # Chromosomes of population
    cost: np.ndarray        # Result of optimization function
    p_selection: np.ndarray # Probaility of selection
    best: np.ndarray        # Chromosomes of current best    

    def __init__(self, nb_population: int, nb_params: int, nb_no_improve: int=0):
        self.X = np.zeros((nb_population, nb_params))
        self.cost = np.zeros((nb_population,))
        self.p_selection = np.zeros((nb_population,))
        self.best = np.zeros((nb_params,))
        self.nb_no_improve = nb_no_improve

    def _reinit(self):
        nb_population = 
        nb_params = 
        self.cost = np.zeros((nb_population,))
        self.p_selection = np.zeros((nb_population,))
        self.best = np.zeros((nb_params,))
        self.nb_no_improve = nb_no_improve



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
        self.best = self.X[1,:]; 

    def do_crossover(self, crossover_fcn: CrossoverFunction_Type):
        # Choose 2 parents
        probability_choose = np.cumsum(self.p_selection)
        for idx_pop in self.X.shape[0]:
            idx_parent_A = np.nonzero(probability_choose>=np.random.rand(1))[0]
            idx_parent_B = np.nonzero(probability_choose>=np.random.rand(1))[0]
            parent_A, parent_B = self.X[idx_parent_A, :], self.X[idx_parent_B, :]

            child = crossover_fcn(parent_A, parent_B)



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


    def run(self, optim_fcn: Callable[[np.ndarray],float], callback_loop: Callable=None):
        '''Execution of ga using optimization function'''

        # Initialization of population
        current_population = Population(self.params.nb_population, self.nb_params)
        current_population.X = self._get_random_population_X()

        # Loop each generation
        for idx_gen in range(self.params.nb_generation_max):

            ### Costs of population
            current_population.update_costs(optim_fcn, self.is_minimization)
            ### Selection
            current_population.do_selection()


            
            
            
