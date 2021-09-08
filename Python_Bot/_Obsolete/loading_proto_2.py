import tkinter as tk
import time
import os
root = tk.Tk()

folder,_ = os.path.split(os.path.realpath(__file__))
file = 'loading.gif'
path_gif = os.path.join(folder, file)
frameCnt = 10
frames = [tk.PhotoImage(file=path_gif, format = f'gif -index {i}') for i in range(0,frameCnt)]

def update(ind):

    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 1
    label.configure(image=frame)
    root.after(100, update, ind)

label = tk.Label(root)
label.pack(fill=tk.BOTH)
root.after(0, update, 0)
root.mainloop()