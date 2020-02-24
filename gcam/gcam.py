from gcam import run_model, run_grad_cam, evaluate_grad_cam


def forward_disabled(model, dataset):
    run_model.run(model, dataset)


def forward_gcam(model, batch):
    run_grad_cam.run(model, batch)


def evaluate_gcam(model, dataset, result_dir, layer='auto'):
    evaluate_grad_cam.evaluate_dataset(model, dataset, result_dir, layer=layer)