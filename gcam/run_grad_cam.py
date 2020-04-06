import numpy as np
from gcam.grad_cam import grad_cam
from gcam.grad_cam.gradcam_utils import *
import inspect


# TODO: DEPRECATED
def run(model, batch, layer='auto', input_key="img"):
    model.eval()
    model_base = type(model).__bases__[0]
    model_GCAM = grad_cam.create_grad_cam(model_base)(model=model)
    # model_GCAM = grad_cam.create_grad_cam(object)(model=model, candidate_layers=[layer])
    model_GBP = grad_cam.create_guided_back_propagation(model_base)(model=model)

    with torch.enable_grad():
        batch[input_key] = torch.tensor(batch[input_key]).unsqueeze(0)
        output = model_GCAM.forward(batch[input_key])
        _ = model_GBP.forward(batch[input_key])
        is_ok = model_GCAM.model.get_ok_list()

        if True in is_ok:  # Only if object are detected
            model_GBP.backward()
            attention_map_GBP = model_GBP.generate()[0]
            model_GCAM.backward()
            attention_map_GCAM = model_GCAM.generate(target_layers=layer)[0]

        image_GCAM = []
        image_GGCAM = []
        batch_size = batch[input_key].shape[0]
        for j in range(batch_size):
            if is_ok[j]:
                map_GCAM_j = attention_map_GCAM[j].squeeze().cpu().numpy()
                map_GBP_j = attention_map_GBP[j].squeeze().cpu().numpy()
                img = batch[input_key][j].squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                image_GCAM.append(generate_gcam(gcam=map_GCAM_j, image=img))
                image_GGCAM.append(generate_guided_gcam(gcam=map_GCAM_j, guided_bp=map_GBP_j))
            else:
                image_GCAM.append(batch[input_key][j])
                image_GGCAM.append(batch[input_key][j])
    return output, np.asarray(image_GCAM), np.asarray(image_GGCAM)