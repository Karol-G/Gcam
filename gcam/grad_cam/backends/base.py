import torch
from torch.nn import functional as F

def create_base_wrapper(base):
    class _BaseWrapper(base):
        """
        Please modify forward() and backward() according to your task.
        """

        def __init__(self, model, is_backward_ready=None, postprocessor=None, retain_graph=False):
            super(_BaseWrapper, self).__init__()
            self.device = next(model.parameters()).device
            self.retain_graph = retain_graph
            self.model = model
            self.handlers = []  # a set of hook function handlers
            if is_backward_ready is None:
                self.is_backward_ready = self.model.is_backward_ready()
            else:
                self.is_backward_ready = is_backward_ready
            self.postprocessor = postprocessor

        def _encode_one_hot(self, ids):
            one_hot = torch.zeros_like(self.logits).to(self.device)
            one_hot.scatter_(1, ids, 1.0)
            return one_hot

        def forward(self, data):
            """
            Simple classification
            """
            self.model.zero_grad()
            self.logits = self.model(data)
            return self.logits

        def backward(self, ids=None, output=None):
            """
            Class-specific backpropagation

            Either way works:
            1. self.logits.backward(gradient=one_hot, retain_graph=True)
            2. (self.logits * one_hot).sum().backward(retain_graph=True)
            """

            if output is not None:
                self.logits = output

            self.logits = self.post_processing(self.postprocessor, self.logits)
            if self.is_backward_ready:
                self.logits.backward(gradient=self.logits, retain_graph=self.retain_graph)
            else:
                if ids is None:
                    ids = self.model.get_category_id_pos()
                # one_hot = self._encode_one_hot(ids)
                one_hot = torch.zeros_like(self.logits).to(self.device)
                for i in range(one_hot.shape[0]):
                    one_hot[i, ids[i]] = 1.0
                self.logits.backward(gradient=one_hot, retain_graph=self.retain_graph)

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

        def generate(self):
            raise NotImplementedError

        def remove_hook(self):
            """
            Remove all the forward/backward hook functions
            """
            for handle in self.handlers:
                handle.remove()

    return _BaseWrapper
