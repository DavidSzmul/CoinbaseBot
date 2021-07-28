import pyautogui as pg
import cv2
import numpy as np
import time

Template_file = 'Mouse_Keyboard/template/template_ETH.png'

# Hide vscode screen
Hide_Pos = (1775, 20)
pg.click(Hide_Pos[0], Hide_Pos[1])
time.sleep(0.1)
screen = pg.screenshot()

#open the main image and convert it to gray scale image
# main_image = cv2.imread(Screen_file)
main_image = np.array(screen)[:, :, ::-1].copy()
gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
#open the template as gray scale image
template = cv2.imread(Template_file, 0)
width, height = template.shape[::-1] #get the width and height
#match the template using cv2.matchTemplate
match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)


# Specify a threshold
threshold = 0.995
# Store the coordinates of matched area in a numpy array
position = np.where(match >= threshold) #get the location of template in the image
if position is None:
    print('No template found')
for point in zip(*position[::-1]): #draw the rectangle around the matched template
   cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 204, 153), 2)

cv2.putText(main_image, str(np.max(match)), point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
cv2.imshow('Template Found', main_image)

# Go to detected zone
pg.moveTo(point[0]+ width/2, point[1]+height/2)
cv2.waitKey(0)