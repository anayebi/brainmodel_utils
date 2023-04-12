def get_base_model_name(model_name):
    if model_name is None:
        return None, False

    base_model_name = model_name
    trained = True
    if "untrained" == model_name.split("_")[0]:
        base_model_name = model_name[len("untrained") + 1 :]
        trained = False

    return base_model_name, trained


def get_model_func_from_name(model_func_dict, model_name, model_kwargs={}):
    base_model_name, _ = get_base_model_name(model_name)
    if base_model_name is None:
        return None

    return model_func_dict[base_model_name](**model_kwargs)


def get_model_transforms_from_name(
    model_transforms_dict, model_name, model_transforms_key="val"
):
    base_model_name, _ = get_base_model_name(model_name)
    if base_model_name is None:
        return None

    return model_transforms_dict[base_model_name][model_transforms_key]


def get_model_path_from_name(model_paths_dict, model_name):
    base_model_name, trained = get_base_model_name(model_name)
    model_path = None
    if trained:
        model_path = model_paths_dict[base_model_name]
    return model_path


def get_model_layers_from_name(model_layers_dict, model_name):
    base_model_name, _ = get_base_model_name(model_name)
    if base_model_name is None:
        return None

    return model_layers_dict[base_model_name]
