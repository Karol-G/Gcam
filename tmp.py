import torch
import numpy as np
import cv2

def method2(a, b, c, d=4, e=5):
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)

def method(*args, **kwargs):
    method2(*args, **kwargs)

method(a=1,b=2,c=3,e=6)