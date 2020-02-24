import torch
import numpy as np

def check(array):
    if not isinstance(array, np.ndarray):
        array = array.detach().cpu().numpy()
    nonzeros = np.count_nonzero(array)
    has_nans = np.isnan(array).any()
    print("Has nans? {}".format(has_nans))
    print("Non zero count: {}".format(nonzeros))