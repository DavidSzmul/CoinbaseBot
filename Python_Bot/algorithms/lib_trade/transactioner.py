from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Transaction:
    '''Dataclass containing all informations about a transaction'''
    success: bool
    name_from: str
    amount_from: float
    name_to: str
    amount_to: float    

    value_from=None
    value_to=None

class Transactioner(ABC):
    '''Class that realize the exchange between to type of money, crypto, trade,...'''

    @abstractmethod
    def convert(self, from_: str, to_: str, value: float) -> Transaction:
        '''Realize and confirm convertion from one trade to another'''

@dataclass
class DefaultTransactioner(Transactioner):
    '''Class that simulate transactions'''

    flag_success: bool=True
    prc_tax: float=0

    def set_flag_success(self, flag_success: bool):
        self.flag_success = flag_success

    def set_prc_tax(self, prc_tax):
        self.prc_tax = prc_tax

    def convert(self, from_: str, to_: str, value: float) -> Transaction:
        '''Simulation of conversion always realized, suppose that value==amount
        to make it simple'''
        return Transaction(self.flag_success, from_, value, to_, value*(1-self.prc_tax))
