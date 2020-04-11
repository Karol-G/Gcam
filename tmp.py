import torch
import numpy as np
import cv2

attention_map = np.full((5,5), 2, dtype=int)
mask = np.full((5,5), 1, dtype=int)
#result1 = cv2.bitwise_and(attention_map, attention_map, mask=mask)
result2 = intersection = attention_map & mask
#print(result1)
print(result2)