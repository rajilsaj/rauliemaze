# Raulie Project

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
import networkx as nx

imagePath = "ok.jpg"
image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (300, 400))

edgesCanny = cv2.Canny(image, 100, 200)
edgesSobel = sobel(image)
edgesSobelUint8 = np.uint8(255 * edgesSobel)
edgesEnsemble = cv2.bitwise_or(edgesCanny, edgesSobelUint8)

def getEdgeCoordinates(edgeImg):
    yCoords, xCoords = np.where(edgeImg > 0)
    return np.column_stack((xCoords, yCoords))

coordsCanny = getEdgeCoordinates(edgesCanny)
coordsSobel = getEdgeCoordinates(edgesSobelUint8)
coordsEnsemble = getEdgeCoordinates(edgesEnsemble)

def buildGraphFromEdges(edgeCoords):
    graph = nx.Graph()
    for (x, y) in edgeCoords:
        graph.add_node((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in graph:
                graph.add_edge((x, y), neighbor)
    return graph

def findNearestPoint(coords, target):
    distances = np.linalg.norm(coords - target, axis=1)
    nearestIdx = np.argmin(distances)
    return tuple(coords[nearestIdx])

def findPath(edgeCoords, startApprox, goalApprox):
    graph = buildGraphFromEdges(edgeCoords)
    start = findNearestPoint(edgeCoords, np.array(startApprox))
    goal = findNearestPoint(edgeCoords, np.array(goalApprox))
    try:
        path = nx.astar_path(graph, start, goal, heuristic=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
        return path
    except nx.NetworkXNoPath:
        return []

def drawPathOnImage(edgeImage, path):
    imgColor = cv2.cvtColor(edgeImage, cv2.COLOR_GRAY2BGR)
    for (x, y) in path:
        if 0 <= y < imgColor.shape[0] and 0 <= x < imgColor.shape[1]:
            imgColor[y, x] = [255, 0, 0]
    return imgColor

desiredStart = (290, 10)
desiredGoal = (290, 390)

pathCanny = findPath(coordsCanny, desiredStart, desiredGoal)
pathSobel = findPath(coordsSobel, desiredStart, desiredGoal)
pathEnsemble = findPath(coordsEnsemble, desiredStart, desiredGoal)

imgCannyWithPath = drawPathOnImage(edgesCanny, pathCanny)
imgSobelWithPath = drawPathOnImage(edgesSobelUint8, pathSobel)
imgEnsembleWithPath = drawPathOnImage(edgesEnsemble, pathEnsemble)

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

axs[0, 0].imshow(edgesCanny, cmap='gray')
axs[0, 0].set_title("Canny - Edges Only")
axs[0, 1].imshow(imgCannyWithPath)
axs[0, 1].set_title("Canny - With Path")

axs[1, 0].imshow(edgesSobelUint8, cmap='gray')
axs[1, 0].set_title("Sobel - Edges Only")
axs[1, 1].imshow(imgSobelWithPath)
axs[1, 1].set_title("Sobel - With Path")

axs[2, 0].imshow(edgesEnsemble, cmap='gray')
axs[2, 0].set_title("Ensemble - Edges Only")
axs[2, 1].imshow(imgEnsembleWithPath)
axs[2, 1].set_title("Ensemble - With Path")

for ax in axs.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()
