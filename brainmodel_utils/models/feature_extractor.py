import copy
import numpy as np
import torch
from ptutils.core.utils import set_seed
from ptutils.models.utils import load_model, load_model_layer
from brainmodel_utils.models import dataloader_utils as dataloaders
from brainmodel_utils.models.utils import get_base_model_name
from brainmodel_utils.core.utils import convert_to_list


class FeatureExtractor:
    """
    Extracts activations from a given layer of a model.
    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain features
        vectorize  : (boolean) whether to convert layer features into vector
        temporal  : (boolean) whether to keep a time dimension
    """

    def __init__(
        self,
        dataloader,
        n_batches=None,
        vectorize=False,
        agg_func=np.concatenate,
        agg_axis=0,
        custom_attr_name="layers",
        temporal=False,
    ):
        self.dataloader = dataloader
        if n_batches is None:
            self.n_batches = len(self.dataloader)
        else:
            self.n_batches = n_batches
        self.vectorize = vectorize
        self.temporal = temporal

        self.agg_func = agg_func
        self.agg_axis = agg_axis
        self.custom_attr_name = custom_attr_name

    def _store_features(self, layer, inp, out):
        out = out.cpu().numpy()

        if self.vectorize:
            if self.temporal:
                self.layer_feats.append(
                    np.reshape(out, (out.shape[0], out.shape[1], -1))
                )
            else:
                self.layer_feats.append(np.reshape(out, (out.shape[0], -1)))
        else:
            self.layer_feats.append(out)

    def extract_features(
        self, model, model_layer,
    ):
        if model_layer != "inputs":
            if torch.cuda.is_available():
                model.cuda().eval()
            else:
                model.cpu().eval()
        else:
            assert model is None

        self.layer_feats = list()
        if (model_layer != "inputs") and (not hasattr(model, self.custom_attr_name)):
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
                    if hasattr(model, self.custom_attr_name):
                        out = getattr(model, self.custom_attr_name)[model_layer]
                        self._store_features(layer=None, inp=None, out=out)

        self.layer_feats = self.agg_func(self.layer_feats, axis=self.agg_axis)
        if model_layer == "inputs" and (not self.vectorize):
            desired_ndim = 4
            axes = (0, 2, 3, 1)
            chf_axis = 1
            if self.temporal:
                desired_ndim = 5
                chf_axis = 2
                axes = (0, 1, 3, 4, 2)
            # batch x h x w x channels (or batch x channels x h x w)
            # if self.temporal, it is batch x time x ...
            assert self.layer_feats.ndim == desired_ndim
            # make it channels last if originally channels first
            if self.layer_feats.shape[chf_axis] == 3:
                self.layer_feats = np.transpose(self.layer_feats, axes=axes)

            assert self.layer_feats.shape[-1] == 3
        elif (not hasattr(model, self.custom_attr_name)) and (model_layer != "inputs"):
            # Reset forward hook so next time function runs, previous hooks are removed
            handle.remove()

        return self.layer_feats


class ModelFeaturesPipeline:
    """
    Pipeline for extracting model features at a given layer.
    Starts from the model name, to loading the dataloader, to calling the FeatureExtractor object.
    """

    def __init__(
        self,
        model_name,
        model_path,
        dataloader_name,
        dataloader_transforms,
        model_layers=None,
        model_kwargs={},
        model_loader_kwargs={},
        model_layer_kwargs={},
        dataloader_kwargs={},
        feature_extractor_kwargs={"vectorize": True},
        seed=0,
        verbose=False,
    ):

        self.model_path = model_path
        self.dataloader_name = dataloader_name
        self.dataloader_transforms = dataloader_transforms
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.model_layer_kwargs = copy.deepcopy(model_layer_kwargs)

        if model_name is not None:
            assert isinstance(model_name, str)
        self.model_name, self.trained = get_base_model_name(model_name)
        if self.trained:
            assert self.model_path is not None
        else:
            assert self.model_path is None
        assert "trained" not in model_loader_kwargs.keys()
        assert "model_path" not in model_loader_kwargs.keys()
        self.model_loader_kwargs = copy.deepcopy(model_loader_kwargs)
        self.model_loader_kwargs["trained"] = self.trained
        self.model_loader_kwargs["model_path"] = self.model_path

        if isinstance(model_layers, str):
            model_layers = convert_to_list(model_layers)
        self.model_layers = model_layers

        self.dataloader_kwargs = copy.deepcopy(dataloader_kwargs)
        self.feature_extractor_kwargs = copy.deepcopy(feature_extractor_kwargs)
        self.seed = seed
        self.verbose = verbose
        self._load_model_from_name()

    def _get_model_func_from_name(self, model_name, model_kwargs):
        """Repo dependent function that returns the model function from the name"""
        raise NotImplementedError

    def _load_model_from_name(self):
        model = self._get_model_func_from_name(
            model_name=self.model_name, model_kwargs=self.model_kwargs
        )
        self.model = load_model(model=model, **self.model_loader_kwargs)

    def _get_model_layers_list(self, model_name, model_kwargs):
        """This function returns the list of model layers to iterate over"""
        raise NotImplementedError

    def _load_layer_from_name(self, model_layer):
        """This function returns the layer module based on its name,
        to be used by the feature extractor"""
        return load_model_layer(
            model=self.model, model_layer=model_layer, **self.model_layer_kwargs
        )

    def _postproc_features(self, features, **kwargs):
        pass

    def get_model_features(self, stimuli, **kwargs):
        # setting seed for untrained models
        set_seed(self.seed)

        self.dataloader = dataloaders.__dict__[self.dataloader_name](
            stimuli,
            dataloader_transforms=self.dataloader_transforms,
            **self.dataloader_kwargs,
        )

        self.feature_extractor = FeatureExtractor(
            dataloader=self.dataloader, **self.feature_extractor_kwargs
        )

        if self.model_layers is None:
            self.model_layers = self._get_model_layers_list(
                model_name=self.model_name, model_kwargs=self.model_kwargs
            )
        assert isinstance(self.model_layers, list)
        if self.model is None:
            # if there is no model function, then the only valid layers for it are the inputs
            assert self.model_layers == ["inputs"]

        layer_feats = dict()
        for curr_layer_name in self.model_layers:
            curr_layer_features = self.feature_extractor.extract_features(
                model=self.model if curr_layer_name != "inputs" else None,
                model_layer=self._load_layer_from_name(curr_layer_name)
                if curr_layer_name != "inputs"
                else "inputs",
            )
            if self.verbose:
                print(
                    f"Current layer: {curr_layer_name}, Activations of shape: {curr_layer_features.shape}"
                )
            curr_layer_features = self._postproc_features(curr_layer_features, **kwargs)
            layer_feats[curr_layer_name] = curr_layer_features

        return layer_feats
