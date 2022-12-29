def get_base_model_name(model_name):
    if model_name is None:
        return None, False

    base_model_name = model_name
    trained = True
    if "untrained" == model_name.split("_")[0]:
        base_model_name = model_name[len("untrained") + 1 :]
        trained = False

    return base_model_name, trained
