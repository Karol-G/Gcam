from gcam.grad_cam import grad_cam
from gcam.grad_cam.gradcam_utils import *
from pathlib import Path

def gcam_hook(model):
    model_base = type(model).__bases__[0]
    return create_gcam_hook(model_base)

def create_gcam_hook(base):
    class GcamHook(base):
        def __init__(self, model, is_backward_ready, output_dir=None, layer='auto', input_key="img", mask_key="gt"):
            super(GcamHook, self).__init__()
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_dir = output_dir + "/" + layer
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir
            self.layer = layer
            self.input_key = input_key
            self.mask_key = mask_key
            self.model = model
            self.model.eval()
            self.model_GCAM = grad_cam.create_grad_cam(object)(model=model, is_backward_ready=is_backward_ready)
            self.counter = 0

        def forward(self, batch):
            with torch.enable_grad():
                if self.input_key is None:
                    input = batch
                else:
                    input = batch[self.input_key]
                output = self.model_GCAM.forward(input)
                self.model_GCAM.backward()
                attention_map_GCAM = self.model_GCAM.generate(target_layer=self.layer)

                batch_size = input.shape[0]
                for j in range(batch_size):
                    map_GCAM_j = attention_map_GCAM[j].squeeze().cpu().numpy()
                    if self.output_dir is not None:
                        save_gcam(filename=self.output_dir + "/attention_map_" + str(self.counter) + ".png", gcam=map_GCAM_j)
                    self.counter += 1
                return output

    return GcamHook