import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt 

## Image Test
imgPath = "../examples/example1.jpg"


def loadMaze(path):
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)
    return binary

def image2Grid(binary):
    return (binary == 255).astype(np.uint8)
    
def solveDisplay(imagePath):
    binary = loadMaze(imagePath)
    #print(binary)
    print(image2Grid(binary))
    return binary
    
plt.imshow(cv2.cvtColor(solveDisplay(imgPath), cv2.COLOR_BGR2RGB))
plt.show()






