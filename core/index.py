import cv2
import numpy as np

# Reading the picture
img1 = cv2.imread('example3.jpg')
img = cv2.imread('example3.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('example3.jpg', )

# Resizing the picture to an upper scale
width = 177*4
height = 221*4

interpolation_attribute =[
    ("area", cv2.INTER_AREA),
    ("nearest", cv2.INTER_NEAREST),
    ("linear", cv2.INTER_LINEAR),
    ("cubic", cv2.INTER_CUBIC),
    ("lanczos4", cv2.INTER_LANCZOS4)
]


#  Optimization of the picture before processing


# Display
col_of_im = np.vstack([img])
#print(col_of_im)

for name, method in interpolation_attribute:
    if name=="area":
        new_size = cv2.resize(col_of_im, (width, height), interpolation=method)
        cv2.imshow(f"{name}", new_size)
        cv2.waitKey(0)


cv2.destroyAllWindows()

