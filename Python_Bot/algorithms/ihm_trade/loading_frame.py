import tkinter as tk
import os

class Loading_Frame(tk.Frame):

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.place(anchor = "center", relx = 0.5, rely = 0.5, relwidth = 1, relheight = 1)

        # GIF
        self.FRAME_CNT = 10
        folder,_ = os.path.split(os.path.realpath(__file__))
        file = 'loading.gif'
        path_gif = os.path.join(folder, file)
        self.frames = [tk.PhotoImage(file=path_gif, format = f'gif -index {i}') for i in range(0,self.FRAME_CNT)]
        self.label = tk.Label(self)
        # Loop variables
        self.idx = 1
        self.id_loop = None

    def start(self):
        frame = self.frames[self.idx]
        self.idx+=1
        if self.idx == self.FRAME_CNT:
            self.idx = 1
        self.label.configure(image=frame)
        self.label.place(anchor = "center", relx = 0.5, rely = 0.5, relwidth = 1, relheight = 1)
        self.id_loop = self.after(100, self.start)

    def stop(self):
        self.after_cancel(self.id_loop)
        self.destroy()

if __name__=="__main__":
    root = tk.Tk()
    load_frame = Loading_Frame(root)
    load_frame.start()
    load_frame.after(2000, load_frame.stop)
    root.mainloop()