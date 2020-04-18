import torch
import numpy as np
import cv2

def method(*args, **kwargs):
    print(args)
    print(kwargs)

method(1, 2, hello=3)
print("OK")
method(1, 2)
print("OK")
method(hello=3)
print("OK")