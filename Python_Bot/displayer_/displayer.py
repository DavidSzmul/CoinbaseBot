from abc import ABC, abstractmethod

class Displayer(ABC):
    '''Abstract Class used to display informations from another class.
    This display can be from different forms:
        -   Using Matlplotlib figures
        -   Using an associated server communicating to an associated Front-End'''

    @abstractmethod
    def reset(self):
        '''Reset displayed informations'''

    @abstractmethod
    def update(self):
        '''Update of displayed informations'''