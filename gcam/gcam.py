import legacy.evaluate_grad_cam
import gcam.gcam_hook


# def forward_disabled(model, dataset, iterations=10):
#     return gcam.run_model.run(model, dataset, iterations=iterations)
#
#
# def forward_gcam(model, batch, layer='auto', input_key="img"):
#     return gcam.run_grad_cam.run(model, batch, layer=layer, input_key=input_key)


def extract(model, dataset, output_dir, layer='auto', input_key="img", mask_key="gt"):
    return legacy.evaluate_grad_cam.extract(model, dataset, output_dir, layer=layer)


def inject(model, output_dir=None, backend="gcam", layer='auto', input_key=None, mask_key=None, postprocessor=None,
           retain_graph=False, dim=2, save_log=False, save_maps=False, save_pickle=False, evaluate=False, evaluation_metric="default", return_score=False, call_dump=False):
    return gcam.gcam_hook.gcam_hook(model)(model, output_dir, backend, layer, input_key, mask_key, postprocessor,
                                           retain_graph, dim, save_log, save_maps, save_pickle, evaluate, evaluation_metric, return_score, call_dump)
