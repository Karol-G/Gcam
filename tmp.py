import torch
import numpy as np
import cv2

image = np.full((5,5), 0, dtype=bool)
mask = np.full((5,5), 255, dtype=bool)
image[0][0] = True
#result = cv2.bitwise_and(image, image, mask=mask)
unique = np.unique(image)
print(unique)