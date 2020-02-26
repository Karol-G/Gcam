import gcam.run_model
import gcam.run_grad_cam
import gcam.evaluate_grad_cam
import gcam.gcam_hook


def forward_disabled(model, dataset, iterations=10):
    return gcam.run_model.run(model, dataset, iterations=iterations)


def forward_gcam(model, batch, layer='auto', input_key="img"):
    return gcam.run_grad_cam.run(model, batch, layer=layer, input_key=input_key)


def evaluate_gcam(model, dataset, result_dir, layer='auto'):
    return gcam.evaluate_grad_cam.evaluate_dataset(model, dataset, result_dir, layer=layer)


def gcam_hook(model):
    return gcam.gcam_hook.gcam_hook(model)