import torch
import gc
from torch.utils.data import DataLoader
from torchviz import make_dot, make_dot_from_trace
import matplotlib.pyplot as plt
import numpy as np

def run(model, dataset, iterations=10):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    outputs = []
    for i, batch in enumerate(data_loader):
        output = model(batch["img"])
        outputs.append(output)
        #print("output: {}".format(output))
        # graph = make_dot(output, params=dict(model.named_parameters()))
        # print(graph)
        if i >= iterations:
            break

    gc.collect()
    torch.cuda.empty_cache()

    return np.asarray(outputs)

if __name__ == "__main__":
    # from models.deepdyn_model import DeepdynModel as Model
    # from models.deepdyn_dataset import DeepdynDataset as Dataset
    from models.unet_seg_model import UnetSegModel as Model
    from models.unet_seg_dataset import UnetSegDataset as Dataset

    DEVICE = "cuda" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(device=DEVICE)
    model = Model(device=DEVICE)

    run(model, dataset)