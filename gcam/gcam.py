import gcam.run_model
import gcam.run_grad_cam
import gcam.evaluate_grad_cam
import gcam.gcam_hook


# def forward_disabled(model, dataset, iterations=10):
#     return gcam.run_model.run(model, dataset, iterations=iterations)
#
#
# def forward_gcam(model, batch, layer='auto', input_key="img"):
#     return gcam.run_grad_cam.run(model, batch, layer=layer, input_key=input_key)


def extract(model, dataset, output_dir, layer='auto', input_key="img", mask_key="gt"):
    return gcam.evaluate_grad_cam.extract(model, dataset, output_dir, layer=layer)


def inject(model, is_backward_ready, output_dir=None, layer='auto', input_key="img", mask_key="gt", postprocessor=None):
    return gcam.gcam_hook.gcam_hook(model)(model, is_backward_ready, output_dir, layer, input_key, mask_key, postprocessor)