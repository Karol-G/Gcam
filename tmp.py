import torch
import numpy as np
import cv2
import DataLoader

class MyCNN():
    def eval(self):
        pass
    def __call__(self, batch):
        return 0

dataset = []


model = MyCNN()
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
# Import gcam
from gcam import gcam
# Inject model with gcam
gcam.inject(model)
# Continue to do what you're doing...
# In this case inference on some new data
model.eval()
for i, batch in enumerate(data_loader):
    output = model(batch)
    # more of your code...



