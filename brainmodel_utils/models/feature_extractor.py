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

    def extract_features(self, model, model_layer):
        assert isinstance(model_layer, str)
        if model_layer != "inputs":
            if torch.cuda.is_available():
                model.cuda().eval()
            else:
                model.cpu().eval()
        else:
            assert model is None
        self.layer_feats = list()
        with torch.no_grad():
            for i, x in enumerate(self.dataloader):
                if i == self.n_batches:
                    break

                print(f"Step {i+1}/{self.n_batches}")
                if torch.cuda.is_available():
                    x = x.cuda()

                if model_layer == "inputs":
                    out = x.cpu().numpy()
                    assert out.ndim == 4
                    if out.shape[1] == 3:
                        out = np.transpose(out, axes=(0, 2, 3, 1))
                    else:
                        assert out.shape[-1] == 3
                else:
                    model(x)
                    out = model.layers[model_layer]

                if not isinstance(out, np.ndarray):
                    out = out.cpu().numpy()

                if self.vectorize:
                    self.layer_feats.append(np.reshape(out, (len(out), -1)))
                else:
                    self.layer_feats.append(out)

        self.layer_feats = np.concatenate(self.layer_feats)
        return self.layer_feats
