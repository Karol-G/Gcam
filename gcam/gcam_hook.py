import torch
from gcam.grad_cam.gradcam_utils import *
from pathlib import Path
import types
import functools
from gcam.grad_cam.backends.guided_backpropagation import create_guided_back_propagation
from gcam.grad_cam.backends.grad_cam import create_grad_cam
from gcam.grad_cam.backends.guided_grad_cam import create_guided_grad_cam
from gcam.grad_cam.backends.grad_cam_pp import create_grad_cam_pp

# TODO: Set requirements in setup.py

def gcam_hook(model):
    # model_base = type(model).__bases__[0]
    return create_gcam_hook(object)

def create_gcam_hook(base):
    class GcamHook(base):
        def __init__(self, model, is_backward_ready, output_dir, backend, layer, input_key, mask_key, postprocessor, retain_graph, dim):
            super(GcamHook, self).__init__()
            self.__dict__ = model.__dict__.copy()
            #torch.backends.cudnn.enabled = False # TODO: out of memory
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                # output_dir = output_dir + "/" + str(layer)
                # Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir
            self.layer = layer
            self.input_key = input_key
            self.mask_key = mask_key
            self.model = model
            self.model.eval()
            self.model_backend, self.heatmap = self._assign_backend(backend, self.model, self.layer, is_backward_ready, postprocessor, retain_graph)
            self.backend = backend
            self.counter = 0
            self.dim = dim
            #print("--------------------SUPER TEST")

        def _assign_backend(self, backend, model, target_layers, is_backward_ready, postprocessor, retain_graph):
            if backend == "gbp":
                return create_guided_back_propagation(object)(model=model, is_backward_ready=is_backward_ready, postprocessor=postprocessor, retain_graph=retain_graph), False
            elif backend == "gcam":
                return create_grad_cam(object)(model=model, target_layers=target_layers, is_backward_ready=is_backward_ready, postprocessor=postprocessor, retain_graph=retain_graph), True
            elif backend == "ggcam":
                return create_guided_grad_cam(object)(model=model, target_layers=target_layers, is_backward_ready=is_backward_ready, postprocessor=postprocessor, retain_graph=retain_graph), False
            elif backend == "gcampp":
                return create_grad_cam_pp(object)(model=model, target_layers=target_layers, is_backward_ready=is_backward_ready, postprocessor=postprocessor, retain_graph=retain_graph), True
            else:
                raise TypeError("Backend does not exist")

        def __call__(self, batch):
            return self.forward(batch)

        def forward(self, batch):
            #print("-------------------------- FORWARD GCAM HOOK --------------------------")
            with torch.enable_grad():
                if self.input_key is None:
                    data_shape = batch.shape[-self.dim:]
                    batch_size = batch.shape[0]
                else:
                    data_shape = batch[self.input_key].shape[-self.dim:]
                    batch_size = batch[self.input_key].shape[0]
                output = self.model_backend.forward(batch, data_shape)
                self.model_backend.backward(output=output)
                attention_map = self.model_backend.generate()

                for layer_name in attention_map.keys():
                    if self.output_dir is not None:
                        if layer_name == "":
                            layer_output_dir = self.output_dir
                        else:
                            layer_output_dir = self.output_dir + "/" + layer_name
                        Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
                    for j in range(batch_size):
                            attention_map_j = attention_map[layer_name][j]
                            if self.output_dir is not None:
                                save_attention_map(filename=layer_output_dir + "/attention_map_" + str(self.counter) + ".png", attention_map=attention_map_j, backend=self.backend)
                            self.counter += 1
                return output

        # TODO: If GcamHook.predict3D(...) gets called, it will be forwarded to model.predict3D(...) and then model.forward(...) wil be called not GcamHook.forward(...)
        # TODO: https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
        # TODO: Maybe replace copy with deepcopy
        # TODO: https://stackoverflow.com/questions/26467564/how-to-copy-all-attributes-of-one-python-object-to-another/26467767
        
        # TODO: Retain graph only with normal backward afterwards, otherwise out of memory
        # TODO: Save list if memory size is 1GB
        # TODO: https://stackoverflow.com/questions/20771470/list-memory-usage

        def __getattr__(self, method):
            def abstract_method(*args):
                #print("-------------------------- ABSTRACT METHOD GCAM HOOK (" + method + ") --------------------------")
                if args == ():
                    return self._copy_func(getattr(self.model, method))(self)
                else:
                    return self._copy_func(getattr(self.model, method))(self, *args)

            return abstract_method

        def _copy_func(self, f):
            #print("-------------------------- COPY FUNC GCAM HOOK --------------------------")
            g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                                   argdefs=f.__defaults__,
                                   closure=f.__closure__)
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = f.__kwdefaults__
            return g

        def _select_ids(self, output):
            # TODO: Implement
            pass

    return GcamHook