import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

## Image Test
imgPath = "../examples/example1.jpg"


def loadMaze(path):
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # ----------- APPLY MASKS TO REMOVE BUNNY AND CARROT ----------
    h, w = img2.shape
    mask_size = 50  # You can adjust this size if needed
    mask_size1 = 20
    # Bunny (top-left corner)
    img2[0:mask_size, 0:mask_size] = 255

    # Carrot (bottom-right corner)
    img2[h-20 - mask_size1:h, w - mask_size1 -20:w] = 255
    # -------------------------------------------------------------

    _, binary = cv2.threshold(img2, 117, 221, cv2.THRESH_BINARY)
    return binary


def image2Grid(binary):
    return (binary == 255).astype(np.uint8)


def solveDisplay(imagePath):
    binary = loadMaze(imagePath)
    print(image2Grid(binary))
    return binary


def apply_a():
    return 0


def detect_start_and_goal():
    return 0


contours, _ = cv2.findContours(loadMaze(imgPath).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        cv2.drawContours(loadMaze(imgPath), [cnt], -1, 0, -1)

cleaned = cv2.bitwise_not(loadMaze(imgPath))
edges = cv2.Canny(cleaned, 177, 221)


plt.imshow(edges, cmap='gray')
plt.imshow(cv2.cvtColor(solveDisplay(imgPath), cv2.COLOR_BGR2RGB))
plt.show()
