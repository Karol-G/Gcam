import numpy as np
from gcam.grad_cam import grad_cam
from gcam.grad_cam.gradcam_utils import *


def run(model, batch, layer='auto'):
    model.eval()
    model_GCAM = grad_cam.GradCAM(model=model)
    # model_GCAM = grad_cam.GradCAM(model=model, candidate_layers=[layer])
    model_GBP = grad_cam.GuidedBackPropagation(model=model)

    with torch.enable_grad():
        batch["img"] = torch.tensor(batch["img"]).unsqueeze(0)
        _ = model_GCAM.forward(batch["img"])
        _ = model_GBP.forward(batch["img"])
        is_ok = model_GCAM.model.get_ok_list()

        if True in is_ok:  # Only if object are detected
            model_GBP.backward()
            attention_map_GBP = model_GBP.generate()[0]
            model_GCAM.backward()
            attention_map_GCAM = model_GCAM.generate(target_layer=layer, dim=2)[0]

        image_GCAM = []
        image_GGCAM = []
        batch_size = batch["img"].shape[0]
        for j in range(batch_size):
            if is_ok[j]:
                map_GCAM_j = attention_map_GCAM[j].squeeze().cpu().numpy()
                map_GBP_j = attention_map_GBP[j].squeeze().cpu().numpy()
                img = batch["img"][j].squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                image_GCAM.append(generate_gcam(gcam=map_GCAM_j, raw_image=img))
                image_GGCAM.append(generate_guided_gcam(gcam=map_GCAM_j, guided_bp=map_GBP_j))
            else:
                image_GCAM.append(batch["img"][j])
                image_GGCAM.append(batch["img"][j])
    return np.asarray(image_GCAM), np.asarray(image_GGCAM)