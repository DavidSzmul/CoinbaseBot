from abc import ABC, abstractmethod

class Transactioner(ABC):
    '''Class that realize the exchange between to type of money, crypto, trade,...'''
    @abstractmethod
    def convert(self, from_: str, to_: str, value: float) -> bool:
        '''Realize and confirm convertion from one trade to another'''

class DefaultTransactioner(Transactioner):
    '''Class that simulate transactions'''

    flag_success: bool=True

    def set_flag_success(self, flag_success: bool):
        self.flag_success = flag_success

    def convert(self, from_: str, to_: str, value: float) -> bool:
        '''Simulation of conversion always realized'''
        return self.flag_success 


