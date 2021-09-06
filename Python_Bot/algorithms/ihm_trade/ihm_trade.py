import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from abc import ABC, abstractmethod
from tkinter.font import Font
from typing import List
import time
import os
import threading

from algorithms.lib_trade.processing_trade import Mode_Algo
import config

class Model_MVC_Trade:
    is_running: bool = False
    mode: Mode_Algo
    def __init__(self):
        self.mode = Mode_Algo.train

    def run(self):
        self.is_running = True
        ctr=0
        while self.is_running:
            ctr+=1
            print(self.mode, ctr)
            time.sleep(1)

    def set_mode(self, mode):
        self.mode = mode

    def stop(self):
        self.is_running = False


class Controller_MVC_Trade:
    def __init__(self, model: Model_MVC_Trade, view, list_Historic_Environement: List):
        self.model = model
        self.view = view
        self.list_Historic_Environement = list_Historic_Environement
    
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

    def get_list_env(self):
        return [d['name'] for d in self.list_Historic_Environement]

    def handle_click_list_env(self, event):
        print(self.view.lb_env.get())

    def handle_search_agent_load(self):
        # Determine Subfoler based on chosen_list
        idx_chosen = self.get_list_env().index(self.view.lb_env.get())
        folder = os.path.join(config.MODELS_DIR, self.list_Historic_Environement[idx_chosen]['subfolder'])
        folder = folder.replace('\\','/')
        
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Choose file on this folder
        file_path = filedialog.askopenfilename(title="Select Model Agent", 
                                                filetypes=[( "Text File" , ".txt" )],
                                                initialdir=folder)
        folder_user, file_user = os.path.split(file_path)

        if file_user=='': # No file chosen
            self.view.edit_agent_load.delete(0,"end")
            return
        if folder_user != folder: # Wrong folder
            raise AssertionError('You must choose a file on the proposed folder')
        self.view.edit_agent_load.delete(0,"end")
        self.view.edit_agent_load.insert(0, file_user)

    def handle_search_agent_save(self):
        # Determine Subfoler based on chosen_list
        idx_chosen = self.get_list_env().index(self.view.lb_env.get())
        folder = os.path.join(config.MODELS_DIR, self.list_Historic_Environement[idx_chosen]['subfolder'])
        folder = folder.replace('\\','/')
        
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Choose file on this folder
        file_path = filedialog.askopenfilename(title="Select Model Agent", 
                                                filetypes=[( "Text File" , ".txt" )],
                                                initialdir=folder)
        folder_user, file_user = os.path.split(file_path)

        if file_user=='': # No file chosen
            self.view.edit_agent_save.delete(0,"end")
            return
        if folder_user != folder: # Wrong folder
            raise AssertionError('You must choose a file on the proposed folder')
        self.view.edit_agent_save.delete(0,"end")
        self.view.edit_agent_save.insert(0, file_user)

    def handle_runstop(self):
        if self.model.is_running:
            self.model.stop()
            self.view.btn_run['text'] = 'Run'
            self.view.frame_params.pack()
            self.view.frame_run.pack_forget()
            self.view.frame_run.pack()
            return

        processThread = threading.Thread(target=self.model.run, args=[])  # <- 1 element list
        processThread.start()
        self.view.btn_run['text'] = 'Stop'
        self.view.frame_params.pack_forget()



class View(ABC):
    @abstractmethod
    def setup(self, controller):
        pass
    
    @abstractmethod
    def start_main_loop(self):
        pass

class TkView_MVC_Trade(View):

    def __init__(self):
        pass

    def _setup_mode(self, controller):
        frame = self.frame_mode
        self.label_mode = tk.Label(frame, text="Mode:")
        self.label_mode.pack()
        self.mode_int_var = tk.IntVar() 
        self.mode_int_var.set(1)
        self.rb_mode = []
        self.rb_mode.append(tk.Radiobutton(frame, text="Train", variable=self.mode_int_var, value=1, command=controller.handle_set_mode))
        self.rb_mode.append(tk.Radiobutton(frame, text="Test", variable=self.mode_int_var, value=2, command=controller.handle_set_mode))
        self.rb_mode.append(tk.Radiobutton(frame, text="Real-Time", variable=self.mode_int_var, value=3, command=controller.handle_set_mode))
        
        for rb in self.rb_mode:
            rb.pack(anchor=tk.W)

    def _setup_env(self, controller):
        # Env is based on the type of historic chosen
        frame = self.frame_env
        self.label_env = tk.Label(frame, text="Environement -> Definition historic:")
        self.label_env.pack()
        self.lb_env = ttk.Combobox(frame, values=controller.get_list_env(), state='readonly', width=100)
        self.lb_env.pack(fill=tk.X, ipadx=5)
        self.lb_env.current(0)
        self.lb_env.bind("<<ComboboxSelected>>", controller.handle_click_list_env)

    def _setup_agent(self, controller):
        # Env is based on the type of historic chosen
        frame = self.frame_agent
        self.label_agent = tk.Label(frame, text="Agent:")
        self.label_agent.pack()
        self.frame_agent_load = tk.Frame(frame)
        self.frame_agent_load.pack()
        self.frame_agent_save = tk.Frame(frame)
        self.frame_agent_save.pack()

        # Load Model
        tk.Label(self.frame_agent_load, text="Load -> ").pack(side = tk.LEFT, ipadx=5)
        self.edit_agent_load = tk.Entry(self.frame_agent_load, bd=2)
        self.edit_agent_load.pack(side = tk.LEFT,fill=tk.X, ipadx=5)

        self.btn_search_agent_load = tk.Button(self.frame_agent_load, text=' ... ',
                                                command=controller.handle_search_agent_load)
        self.btn_search_agent_load.pack(side = tk.RIGHT, ipadx=10)

        # Save Model
        tk.Label(self.frame_agent_save, text="Save -> ").pack(side = tk.LEFT, ipadx=5)
        self.edit_agent_save = tk.Entry(self.frame_agent_save, bd=2)
        self.edit_agent_save.pack(side = tk.LEFT,fill=tk.X, ipadx=5)

        self.btn_search_agent_save = tk.Button(self.frame_agent_save, text=' ... ',
                                                command=controller.handle_search_agent_save)
        self.btn_search_agent_save.pack(side = tk.RIGHT, ipadx=10)

    def _setup_runstop(self, controller):
        # Env is based on the type of historic chosen
        frame = self.frame_run
        self.btn_run = tk.Button(frame, text='Run', command=controller.handle_runstop, width=30, height=30, font=Font(self.root,size=12))
        self.btn_run.pack(pady=20)
 
        
    def setup(self, controller):

        # setup tkinter
        self.root = tk.Tk()
        self.root.geometry("400x200")
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
        self.frame_run = tk.Frame(self.frame)
        self.frame_run.pack(side=tk.TOP)

        self._setup_mode(controller)
        self._setup_env(controller)
        self._setup_agent(controller)
        self._setup_runstop(controller)

    
    def start_main_loop(self):
        # start the loop
        self.root.mainloop()

if __name__=="__main__":
    # create the MVC & start the application
    list_Historic_Environement = [
        {'name': 'step x5 every 50 cycles on 5 days',
        'subfolder': 'x5_50cycles_5d',
        'min_historic': [1, 5, 25, 125],
        'nb_cycle_historic': [50, 50, 50, 50],
        },
        {'name': 'step x2 every 15 cycles on 1 days',
        'subfolder': 'x2_15cycles_5d',
        'min_historic': [1, 2, 4, 8],
        'nb_cycle_historic': [15, 15, 15, 15],
        }
    ]
    c = Controller_MVC_Trade(Model_MVC_Trade(), TkView_MVC_Trade(), list_Historic_Environement)
    c.start()