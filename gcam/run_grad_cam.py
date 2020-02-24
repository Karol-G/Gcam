import torch
from gcam.grad_cam import grad_cam
from gcam.grad_cam.gradcam_utils import *


def run(model, batch):
    model.eval()
    layer = 'auto'
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

        if is_ok[0]:
            image_GCAM = save_gcam(gcam=attention_map_GCAM, filepath=batch["filepath"])
            image_GGCAM = save_guided_gcam(guided_gcam=torch.mul(attention_map_GCAM, attention_map_GBP))
            return image_GCAM, image_GGCAM
        else:
            return batch["img"], batch["img"]