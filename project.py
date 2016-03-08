import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

state = 0
path = []
r = 0
g = 0
b = 0

#0 = neutral
#1 = drawing
#2 = erasing

def draw(gesture, location, mask):
    x,y = location

    if len(path) >= 10:
        del path[0]

    if x > 200:
        if 0 < y < 50:
            state = 1
        elif 75 < y < 125:
            state = 2
        else:
            state = 0

    if x < 50:
        if 0 < y < 50:
            r = 255
        elif 75 < y < 125:
            g = 255
        else:
            r = 0
            b = 0
            g = 0

    if state = 1:
        cv2.circle(mask, location, 5, (b,g,r), -1)
    elif state = 2:
        cv2.circle(mask, location, 10, (0,0,0), -1)

    return

def main():

    return 

if __name__ == '__main__':
  main()
