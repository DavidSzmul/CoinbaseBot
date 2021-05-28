import pyautogui as pg
import time
 
def delete_for_everyone():
    pg.click(807, 979)
    pg.typewrite("hello")
    pg.typewrite(["enter"])
    time.sleep(2)
    pg.click(1621, 896)
    pg.click(1621, 896)
     
    # time.sleep(1)
    pg.click(1693, 859)
     
    # time.sleep(1)
    pg.click(1014, 669)
     
    # time.sleep(1)
    pg.click(1111, 605)
     
a=20
time.sleep(10)
while(a!=0):
    delete_for_everyone()
    a=a-1