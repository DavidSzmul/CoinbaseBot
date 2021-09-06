import tkinter as tk
from abc import ABC, abstractmethod
from typing import List
from algorithms.lib_trade.processing_trade import Mode_Algo
import time

class Model_MVC_Trade:
    end_of_loop: bool
    mode: Mode_Algo
    def __init__(self, main_trade_algo):

        self.end_of_loop=False
        self.mode = Mode_Algo.train

    def run(self,item):
        while not self.end_of_loop:
            print('Loop')
            time.sleep(1)
        self.end_of_loop = False

    def set_mode(self, mode):
        self.mode = mode

    def stop(self):
        self.end_of_loop=True


class Controller_MVC_Trade:
    def __init__(self, model: Model_MVC_Trade, view):
        self.model = model
        self.view = view
    
    def start(self):
        self.view.setup(self)
        self.view.start_main_loop()

    def handle_set_mode(self):
        mode_int = self.view.mode_int_var.get()
        if mode_int==1:
            self.model.set_mode(Mode_Algo.train)
        elif mode_int==2:
            self.model.set_mode(Mode_Algo.test)
        elif mode_int==3:
            self.model.set_mode(Mode_Algo.real_time)
        print('Set', self.model.mode)

    def handle_click_list_env(self, event):
        i = self.view.lb_env.curselection()
        print(self.view.lb_env.get(i))


class View(ABC):
    @abstractmethod
    def setup(self, controller):
        pass
    
    @abstractmethod
    def start_main_loop(self):
        pass

class TkView_MVC_Trade(View):

    def __init__(self, list_type_historic: List):
        self.list_type_historic = list_type_historic

    def _setup_mode(self, controller):
        frame = self.frame_mode
        self.label_mode = tk.Label(frame, text="Mode:")
        self.label_mode.pack()
        self.mode_int_var = tk.IntVar() 
        self.rb_mode = []
        self.rb_mode.append(tk.Radiobutton(frame, text="Train", variable=self.mode_int_var, value=1, command=controller.handle_set_mode))
        self.rb_mode.append(tk.Radiobutton(frame, text="Test", variable=self.mode_int_var, value=2, command=controller.handle_set_mode))
        self.rb_mode.append(tk.Radiobutton(frame, text="Real-Time", variable=self.mode_int_var, value=3, command=controller.handle_set_mode))
        for rb in self.rb_mode:
            rb.pack(anchor=tk.W)

    def _setup_env(self, controller):
        # Env is based on the type of historic chosen
        frame = self.frame_env
        self.label_env = tk.Label(frame, text="Historic Definition:")
        self.label_env.pack()
        self.lb_env = tk.Listbox(frame)
        self.lb_env.pack(fill=tk.BOTH, expand=1)
        self.lb_env.bind('<<ListboxSelect>>', controller.handle_click_list_env)

        for i,type in enumerate(self.list_type_historic):
            self.lb_env.insert(i, type)
        self.lb_env.select_set(0)
        
    def setup(self, controller):

        # setup tkinter
        self.root = tk.Tk()
        self.root.geometry("400x400")
        self.root.title("Coinbase Bot")

        # create the gui
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=1)

        # Frames to organize ihm
        self.frame_params = tk.Frame(self.frame)
        self.frame_params.pack(side=tk.TOP)
        self.frame_mode = tk.Frame(self.frame_params)
        self.frame_mode.pack(side=tk.LEFT)
        self.frame_agent_env = tk.Frame(self.frame_params)
        self.frame_agent_env.pack(side=tk.RIGHT)
        self.frame_env = tk.Frame(self.frame_agent_env)
        self.frame_env.pack(side=tk.TOP)
        self.frame_agent = tk.Frame(self.frame_agent_env)
        self.frame_agent.pack(side=tk.TOP)

        self._setup_mode(controller)
        self._setup_env(controller)

        # self.label = tk.Label(self.frame, text="Result:")
        # self.label.pack()
        # self.list = tk.Listbox(self.frame)
        # self.list.pack(fill=tk.BOTH, expand=1)
        # self.generate_uuid_button = tk.Button(self.frame, text="Generate UUID", command=controller.handle_click_generate_uuid)
        # self.generate_uuid_button.pack()
        # self.clear_button = tk.Button(self.frame, text="Clear list", command=controller.handle_click_clear_list)
        # self.clear_button.pack()

    
    def start_main_loop(self):
        # start the loop
        self.root.mainloop()

if __name__=="__main__":
    # create the MVC & start the application
    c = Controller_MVC_Trade(Model_MVC_Trade(None), TkView_MVC_Trade(['Hello', 'Yo']))
    c.start()