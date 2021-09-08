from enum import Enum
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.font import Font
from tkinter.filedialog import asksaveasfile

from abc import ABC, abstractmethod
from typing import  Any, List
import time
import os
import threading

import config
from algorithms.ihm_trade.loading_frame import Loading_Frame


class Abstract_RL_App(ABC):
    '''Abstraction of App'''

    is_running: bool=False


    @abstractmethod
    def save_model(self, path:str):
        '''Save of model agent'''

    @abstractmethod
    def define_params(self, min_historic: List[int],nb_cycle_historic: List[int], path_agent:str):
        '''Definition of environment + agent'''

    @abstractmethod
    def update_train_test_dtb(self):
        '''Update all properties about database (refresh more recent trade values)'''

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
    
    # @abstractmethod
    # def real_time(self):
    #     pass

    def stop(self):
        self.is_running = False


# def _use_thread(func, self, *args):
#         if self.is_running:
#             return
#         self.is_running = True
#         processThread = threading.Thread(target=func, args=[self]+args)
#         processThread.start()

class Easy_RL_App(Abstract_RL_App):
    '''Simulation of more complex application (for testing)'''

    path_agent: str
    
    def _thread_run(func):
        def inner(self):
            if self.is_running:
                return
            self.is_running = True
            processThread = threading.Thread(target=func, args=[self])
            processThread.start()
        return inner

    @_thread_run
    def train(self):
        ctr=0
        while self.is_running:
            ctr+=1
            print('Train:', ctr)
            time.sleep(1)

    @_thread_run
    def test(self):
        ctr=0
        while self.is_running:
            ctr+=1
            print('Test:', ctr)
            time.sleep(0.5)


    def update_train_test_dtb(self):
        time.sleep(2)
        print('Database updated: Ready to execute')

    def define_params(self, min_historic: List[int],nb_cycle_historic: List[int], path_agent:str):
        '''Definition of environment + agent'''
        print('Params Initialized')
        self.update_train_test_dtb()

    def save_model(self, path: str):
        print('Agent Model Saved\n', path)

class Enum_Frames(Enum):
    Params_Frame = 1
    Run_Frame = 2
    Stop_Frame = 3

class Controller_MVC_Trade:
    
    model: Abstract_RL_App
    view: Any
    default_folder: str
    is_waiting_model: bool=False

    def __init__(self, model: Abstract_RL_App, view, list_Historic_Environement: List):
        self.model = model
        self.view = view
        self.list_Historic_Environement = list_Historic_Environement
    
    def start(self, first_frame: Enum_Frames=Enum_Frames.Params_Frame):
        self.view.setup(self, first_frame=first_frame)
        self.update_default_folder()
        self.view.start_main_loop()

    def _thread_wait_model(func):
        def inner(self):
            if self.is_waiting_model:
                return
            self.is_waiting_model = True
            self.view.add_loading_frame()

            def close_callback():
                self.view.delete_loading_frame()
                self.is_waiting_model = False
                
            def wrapper():
                func(self)
                close_callback()
                
            processThread = threading.Thread(target=wrapper, args=[])
            processThread.start()
        return inner


    def get_list_env(self):
        return [d['name'] for d in self.list_Historic_Environement]

    def _set_agent_load_file(self, file: str):
        self.view.edit_agent_load.configure(state='normal')
        self.view.edit_agent_load.delete(0,"end")
        self.view.edit_agent_load.insert(0, file)
        self.view.edit_agent_load.configure(state='readonly')

    def handle_click_list_env(self, event):
        print(self.view.lb_env.get())
        self.update_default_folder()
        self._set_agent_load_file('')


    def update_default_folder(self):
        if not hasattr(self.view, 'lb_env'):
            self.default_folder = ''
            return
        idx_chosen = self.get_list_env().index(self.view.lb_env.get())
        self.default_folder = os.path.join(config.MODELS_DIR, self.list_Historic_Environement[idx_chosen]['subfolder']).replace('\\','/')
        if not os.path.exists(self.default_folder):
            os.makedirs(self.default_folder)

    def handle_search_load_agent(self):
        # Choose file on this folder
        file_path = filedialog.askopenfilename(title="Select Model Agent", 
                                                filetypes=[( "Text File" , ".txt" )],
                                                initialdir=self.default_folder)
        
        folder_user, file_user = os.path.split(file_path)
        if (folder_user != self.default_folder and file_user !=''): # Wrong folder
            self._set_agent_load_file('')
            raise AssertionError('You must choose a file on the proposed folder')
        self._set_agent_load_file(file_user)

    @_thread_wait_model
    def handle_confirm_params(self):
        # Environment Definition
        idx_env_chosen = self.get_list_env().index(self.view.lb_env.get())
        min_historic = self.list_Historic_Environement[idx_env_chosen]['min_historic']
        nb_cycle_historic = self.list_Historic_Environement[idx_env_chosen]['nb_cycle_historic']

        # Agent Path
        file = self.view.edit_agent_load.get()
        if file == '':
            path_agent = None
        else:
            folder = os.path.join(config.MODELS_DIR, self.list_Historic_Environement[idx_env_chosen]['subfolder'])
            path_agent = os.path.join(folder, file)

        # Set parameters to app
        self.model.define_params(min_historic, nb_cycle_historic, path_agent)
        self.view.set_new_frame(Enum_Frames.Run_Frame, self)

    @_thread_wait_model
    def handle_update_dtb(self):
        self.model.update_train_test_dtb()

    def handle_return(self):
        self.view.set_new_frame(Enum_Frames.Params_Frame, self)

    def handle_train(self):
        self.model.train()
        self.view.set_new_frame(Enum_Frames.Stop_Frame, self)

    def handle_test(self):
        self.model.test()
        self.view.set_new_frame(Enum_Frames.Stop_Frame, self)

    def handle_save(self):
        # Determine path of save
        files = [('Text Document', '*.txt')]
        res = asksaveasfile(filetypes = files, defaultextension = files, initialdir=self.default_folder)
        if not res: # Canceling
            return
        
        path_user = res.name
        folder_user, _ = os.path.split(path_user)
        if (folder_user != self.default_folder): # Wrong folder
            raise AssertionError('You must determine a file on the proposed folder')
        self.model.save_model(path_user)

    def handle_stop(self):
        self.model.stop()
        self.view.set_new_frame(Enum_Frames.Run_Frame, self)



class View(ABC):
    @abstractmethod
    def setup(self, controller):
        pass
    
    @abstractmethod
    def start_main_loop(self):
        pass

class TkView_MVC_Trade(View):

    frame: tk.Tk=None

    def set_new_frame(self, enum: Enum_Frames, controller: Controller_MVC_Trade):

        dict_frame = {
            Enum_Frames.Params_Frame: self._setup_frame_params,
            Enum_Frames.Run_Frame: self._setup_run,
            Enum_Frames.Stop_Frame: self._setup_stop,
        }
        if enum not in dict_frame:
            raise ValueError('enum must be in dictionary')
        if self.frame is not None:
            self.frame.destroy()
        setup_fcn = dict_frame[enum]
        self.frame = setup_fcn(self.container, controller)
        self.frame.tkraise()

    def _setup_frame_params(self, container: tk.Frame, controller: Controller_MVC_Trade) -> tk.Tk:
        '''Setup parameters linked to App'''
        frame_params = tk.Frame(container)
        frame_params.pack(side=tk.TOP, ipady=10)

        title = tk.Label(frame_params, text="Definition of Parameters", font=Font(size=16))
        title.pack(ipady=10)

        # Environement
        frame_env = tk.Frame(frame_params)
        frame_env.pack(ipady=5)
        self.label_env = tk.Label(frame_env, text="Environement -> Definition historic:")
        self.label_env.pack()
        self.lb_env = ttk.Combobox(frame_env, values=controller.get_list_env(), state='readonly', width=50)
        self.lb_env.pack(ipadx=5)
        self.lb_env.current(0)
        self.lb_env.bind("<<ComboboxSelected>>", controller.handle_click_list_env)

        # Agent
        frame_agent = tk.Frame(frame_params)
        frame_agent.pack(ipady=5)
        tk.Label(frame_agent, text="Agent:").pack()
        tk.Label(frame_agent, text="Load -> ").pack(side = tk.LEFT, ipadx=5)
        self.edit_agent_load = tk.Entry(frame_agent, bd=2, state='readonly')
        self.edit_agent_load.pack(side = tk.LEFT,fill=tk.X, ipadx=5)

        self.btn_search_agent = tk.Button(frame_agent, text=' ... ',
                                                command=controller.handle_search_load_agent)
        self.btn_search_agent.pack(side = tk.RIGHT)

        # Confirm Button
        self.btn_confirm_params = tk.Button(frame_params, text='CONFIRM', font=Font(size=12),
                                                command=controller.handle_confirm_params)
        self.btn_confirm_params.pack(ipady=10)

        return frame_params


    def _setup_run(self, container: tk.Frame, controller: Controller_MVC_Trade) -> tk.Frame:
        '''Setup Tasks linked to App'''
        frame_tasks = tk.Frame(container)
        frame_tasks.pack(side=tk.TOP, ipady=10, fill=tk.X)

        title = tk.Label(frame_tasks, text="TASKS", font=Font(size=16))
        title.pack(ipady=10)
        
        frame_right = tk.Frame(frame_tasks)
        frame_right.pack(side=tk.LEFT, ipadx=5, ipady=5)

        # Button Update + Return
        frame_left = tk.Frame(frame_tasks)
        frame_left.pack(side=tk.LEFT, ipadx=5, ipady=5)
        self.btn_update_dtb = tk.Button(frame_left, text='Update Database',
                                                command=controller.handle_update_dtb)
        self.btn_update_dtb.pack()
        self.btn_return = tk.Button(frame_left, text='<- Return', font=Font(size=8),
                                                command=controller.handle_return)
        self.btn_return.pack(side=tk.BOTTOM)

        # Train / Test / Save
        WIDTH_CENTER = 10
        PADDING_CENTER = 5
        frame_center = tk.Frame(frame_tasks)
        frame_center.pack(side=tk.LEFT, ipadx=PADDING_CENTER, ipady=PADDING_CENTER)

        self.btn_train = tk.Button(frame_center, text='Train', width=WIDTH_CENTER,
                                                command=controller.handle_train)
        self.btn_train.pack(padx=PADDING_CENTER, pady=PADDING_CENTER)

        self.btn_test = tk.Button(frame_center, text='Test', width=WIDTH_CENTER,
                                                command=controller.handle_test)
        self.btn_test.pack(padx=PADDING_CENTER, pady=PADDING_CENTER)

        self.btn_save = tk.Button(frame_center, text='Save', width=WIDTH_CENTER,
                                                command=controller.handle_save)
        self.btn_save.pack(padx=PADDING_CENTER, pady=PADDING_CENTER) 
        return frame_tasks    
    
    def _setup_stop(self, container: tk.Frame, controller: Controller_MVC_Trade) -> tk.Frame:
        '''Setup Tasks linked to App'''
        frame_stop = tk.Frame(container)
        frame_stop.pack(side=tk.TOP, ipady=10, fill=tk.BOTH)

        title = tk.Label(frame_stop, text="Running...", font=Font(size=16))
        title.pack(ipady=10)

        self.btn_stop = tk.Button(frame_stop, text='Stop', font=Font(size=16),
                                                command=controller.handle_stop)
        self.btn_stop.pack(fill=tk.BOTH)
        return frame_stop   
    
        
    def setup(self, controller: Controller_MVC_Trade, first_frame: Enum_Frames=Enum_Frames.Params_Frame):
        # setup tkinter
        self.root = tk.Tk()
        self.root.geometry("400x250")
        
        self.root.title("Coinbase Bot")
        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=1)
        # Show first Frame
        self.set_new_frame(first_frame, controller)     

    def add_loading_frame(self):
        self.loading_frame = Loading_Frame(self.container)
        self.loading_frame.start()

    def delete_loading_frame(self):
        if self.loading_frame:
            self.loading_frame.stop()
        
       
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

    model = Easy_RL_App() # Corresponds to the application to execute
    view = TkView_MVC_Trade() # WIDGETS
    c = Controller_MVC_Trade(model, view, list_Historic_Environement)
    c.start(first_frame=Enum_Frames.Run_Frame)