import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt 

## Image Test
imgPath = r"C:\Users\admin\Documents\www\RaulieMaze\mazes\example3.png"


def loadMaze(path):
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        print(f"Error: Could not load image at {path}")
        return None
    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(img2, (5, 5), 0) # Kernel size 5x5, sigmaX 0 (auto)
    # Use Otsu's thresholding for automatic threshold determination
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def image2Grid(binary, invert=False):
    # Define a kernel for morphological operations
    kernel = np.ones((3,3), np.uint8) # 3x3 square kernel

    # Apply morphological operations to clean up the binary image
    # Erosion followed by Dilation (Opening) can remove small objects/noise
    # Dilation followed by Erosion (Closing) can fill small holes/gaps
    # We'll try a combination that generally works well for mazes:
    # First, a small erosion to remove noise, then dilation to connect paths
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Now convert the cleaned binary image to the grid
    if invert:
        grid = (dilated == 0).astype(np.uint8)
    else:
        grid = (dilated == 255).astype(np.uint8)

    # Print a small portion of the grid for debugging
    print("Sample of maze grid (top-left 5x5):")
    print(grid[:5, :5])
    return grid
    
def solveDisplay(imagePath):
    binary = loadMaze(imagePath)
    #print(binary)
    print(image2Grid(binary))
    return binary
    
# plt.imshow(cv2.cvtColor(solveDisplay(imgPath), cv2.COLOR_BGR2RGB))
# plt.show()

def bfs(grid, start, end):
    rows, cols = grid.shape
    queue = deque([(start, [start])])  # (current_position, path_to_current_position)
    visited = set([start])

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == end:
            return path

        # Define possible movements (up, down, left, right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check if the new position is within bounds and is a valid path (0 for path, 1 for wall)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None  # No path found

if __name__ == "__main__":
    maze_binary = loadMaze(imgPath)

    # Visualize the binary image
    plt.imshow(maze_binary, cmap='gray')
    plt.title("Binary Maze Image (after thresholding)")
    plt.axis('off') # Turn off axes
    save_path_binary = "binary_maze.png"
    print(f"Saving binary maze image to: {save_path_binary}")
    plt.savefig(save_path_binary)
    # plt.show() # Commented out to prevent display

    maze_grid = image2Grid(maze_binary, invert=True)

    # Define start and end points (adjust based on your maze image)
    # Assuming start is top-left (0,0) and end is bottom-right
    start_point = (0, 0)
    end_point = (maze_grid.shape[0] - 1, maze_grid.shape[1] - 1)

    print(f"Solving maze from {start_point} to {end_point}...")

    # Visualize the maze grid
    plt.imshow(maze_grid, cmap='gray')
    plt.title("Maze Grid")
    plt.show()

    path = bfs(maze_grid, start_point, end_point)

    if path:
        print("Path found! Length:", len(path))
        # Create a color version of the maze to draw the path
        maze_display = cv2.cvtColor(maze_binary, cv2.COLOR_GRAY2BGR)
        
        # Draw the path in orange (BGR: 0, 165, 255)
        for i in range(len(path) - 1):
            pt1 = (path[i][1], path[i][0]) # (col, row) for OpenCV
            pt2 = (path[i+1][1], path[i+1][0])
            cv2.line(maze_display, pt1, pt2, (0, 165, 255), 2) # Thickness 2

        plt.imshow(cv2.cvtColor(maze_display, cv2.COLOR_BGR2RGB))
        plt.title("Maze with BFS Path")
        plt.axis('off') # Turn off axes
        save_path_solved = "solved_maze.png"
        print(f"Saving solved maze image to: {save_path_solved}")
        plt.savefig(save_path_solved)
        # plt.show() # Commented out to prevent display
    else:
        print("No path found.")