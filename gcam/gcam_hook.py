from gcam.grad_cam import grad_cam
from gcam.grad_cam.gradcam_utils import *
from pathlib import Path
from torch.nn import functional as F
import types
import functools

# TODO: Set requirements in setup.py

def gcam_hook(model):
    # model_base = type(model).__bases__[0]
    return create_gcam_hook(object)

def create_gcam_hook(base):
    class GcamHook(base):
        def __init__(self, model, is_backward_ready, output_dir, layer, input_key, mask_key, postprocessor):
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
            self.postprocessor = postprocessor
            #model_base = type(model).__bases__[0]
            #self.model = create_model_wrapper(model_base)(model, postprocessor)
            self.model_GCAM = grad_cam.create_grad_cam(object)(model=self.model, target_layers=self.layer, is_backward_ready=is_backward_ready)
            self.counter = 0
            print("--------------------SUPER TEST")

        def __call__(self, batch):
            return self.forward(batch)

        def forward(self, batch):
            print("-------------------------- FORWARD GCAM HOOK --------------------------")
            with torch.enable_grad():
                input = batch
                if self.input_key is None:
                    #input = batch
                    image_shape = batch.shape[2:]
                    batch_size = input.shape[0]
                else:
                    #input = batch[self.input_key]
                    image_shape = batch[self.input_key].shape[2:]
                    batch_size = input[self.input_key].shape[0]
                output = self.model_GCAM.forward(input, image_shape)
                output = self.post_processing(self.postprocessor, output)
                self.model_GCAM.backward(output=output)
                attention_map_GCAM = self.model_GCAM.generate()

                for layer_name in attention_map_GCAM.keys():
                    if self.output_dir is not None:
                        layer_output_dir = self.output_dir + "/" + layer_name
                        Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
                    for j in range(batch_size):
                            map_GCAM = attention_map_GCAM[layer_name][j]
                            #map_GCAM_j = attention_map_GCAM[j].squeeze().cpu().numpy()
                            if self.output_dir is not None:
                                save_gcam(filename=layer_output_dir + "/attention_map_" + str(self.counter) + ".png", gcam=map_GCAM)
                            self.counter += 1
                return output

        # TODO: If GcamHook.predict3D(...) gets called, it will be forwarded to model.predict3D(...) and then model.forward(...) wil be called not GcamHook.forward(...)
        # TODO: https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
        # TODO: Maybe replace copy with deepcopy
        # TODO: https://stackoverflow.com/questions/26467564/how-to-copy-all-attributes-of-one-python-object-to-another/26467767
        
        # TODO: Retain graph only with normal backward afterwards, otherwise out of memory
        # TODO: Save list if memory size is 1GB
        # TODO: https://stackoverflow.com/questions/20771470/list-memory-usage
        # def __getattr__(self, method):
        #     def abstract_method(*args):
        #         print("-------------------------- ABSTRACT METHOD GCAM HOOK (" + method + ") --------------------------")
        #         if args==():
        #             return getattr(self.model, method)()
        #         else:
        #             return getattr(self.model, method)(args)
        #
        #     return abstract_method

        def __getattr__(self, method):
            def abstract_method(*args):
                print("-------------------------- ABSTRACT METHOD GCAM HOOK (" + method + ") --------------------------")
                if args == ():
                    return self.copy_func(getattr(self.model, method))(self)
                else:
                    return self.copy_func(getattr(self.model, method))(self, *args)

            return abstract_method

        def copy_func(self, f):
            print("-------------------------- COPY FUNC GCAM HOOK --------------------------")
            g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                                   argdefs=f.__defaults__,
                                   closure=f.__closure__)
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = f.__kwdefaults__
            return g

        def post_processing(self, postprocessor, output):
            if postprocessor is None:
                return output
            elif postprocessor == "sigmoid":
                output = torch.sigmoid(output)
            elif postprocessor == "softmax":
                output = F.softmax(output, dim=1)
            else:
                output = postprocessor(output)
            return output

        def select_ids(self, output):
            # TODO: Implement
            pass

    return GcamHook

# def create_model_wrapper(base):
#     class ModelWrapper(base):
#         def __init__(self, model, postprocessor):
#             super(ModelWrapper, self).__init__()
#             self.model = model
#             self.postprocessor = postprocessor
#
#         def forward(self, batch):
#             output = self.model(batch)
#             return self.post_processing(self.postprocessor, output)
#
#         def post_processing(self, postprocessor, output):
#             if postprocessor is None:
#                 return output
#             elif postprocessor == "sigmoid":
#                 output = torch.sigmoid(output)
#             elif postprocessor == "softmax":
#                 output = F.softmax(output, dim=1)
#             else:
#                 output = postprocessor(output)
#             return output
#
#     return ModelWrapper