from gcam import run_model, run_grad_cam, evaluate_grad_cam


def forward_disabled(model, dataset, iterations=10):
    return run_model.run(model, dataset, iterations=iterations)


def forward_gcam(model, batch, layer='auto', input_key="img"):
    return run_grad_cam.run(model, batch, layer=layer, input_key=input_key)


def evaluate_gcam(model, dataset, result_dir, layer='auto'):
    return evaluate_grad_cam.evaluate_dataset(model, dataset, result_dir, layer=layer)