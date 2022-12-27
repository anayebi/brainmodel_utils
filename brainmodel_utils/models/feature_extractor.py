import numpy as np
import torch


class FeatureExtractor:
    """
    Extracts activations from a layer of a model.
    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
    """

    def __init__(self, dataloader, n_batches=None, vectorize=False):
        self.dataloader = dataloader
        if n_batches is None:
            self.n_batches = len(self.dataloader)
        else:
            self.n_batches = n_batches
        self.vectorize = vectorize

    def _store_features(self, layer, inp, out):
        out = out.cpu().numpy()

        if self.vectorize:
            self.layer_feats.append(np.reshape(out, (len(out), -1)))
        else:
            self.layer_feats.append(out)

    def extract_features(self, model, model_layer, agg_func=np.concatenate, axis=0):
        if model_layer != "inputs":
            if torch.cuda.is_available():
                model.cuda().eval()
            else:
                model.cpu().eval()
        else:
            assert model is None
        self.layer_feats = list()
        if model_layer != "inputs":
            # Set up forward hook to extract features
            handle = model_layer.register_forward_hook(self._store_features)

        with torch.no_grad():
            for i, x in enumerate(self.dataloader):
                if i == self.n_batches:
                    break

                print(f"Step {i+1}/{self.n_batches}")
                if torch.cuda.is_available():
                    if isinstance(x, dict):
                        for k, v in x.items():
                            x[k] = v.cuda()
                    else:
                        x = x.cuda()

                if model_layer == "inputs":
                    assert not isinstance(x, dict)
                    self._store_features(layer=None, inp=None, out=x)
                else:
                    model(x)

        self.layer_feats = agg_func(self.layer_feats, axis=axis)
        if model_layer == "inputs":
            # batch x h x w x channels (or batch x channels x h x w)
            assert self.layer_feats.ndim == 4
            # make it channels last if originally channels first
            if self.layer_feats.shape[1] == 3:
                self.layer_feats = np.transpose(self.layer_feats, axes=(0, 2, 3, 1))

            assert self.layer_feats.shape[-1] == 3
        else:
            # Reset forward hook so next time function runs, previous hooks are removed
            handle.remove()

        return self.layer_feats
