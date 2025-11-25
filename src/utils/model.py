
import timm

def get_model(model_name):
    """
    Load a model.

    Currently, only the models available in the package timm
    are supported.
    """

    if timm.is_model(model_name):
        model = timm.create_model(model_name, True)
    else:
        raise ValueError(f"The '{model_name}' is not supported.")
    return model