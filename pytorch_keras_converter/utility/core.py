from .torch2keras import convert_torch2keras
from .torch2keras import convert_torch2keras_file
from .LayerRepresentation import normalizeShape


def convert_to_file(model, framework):
    """
    Converts a models between PyTorch and Keras

    Arguments:
        -model:
            the model to convert
        -framework:
            the framework created python file should use

    Returns:
        the model (LayerRepresentation)

    Raises:
        -NotImplementedError:
            if the conversion isn't supported yet
    """

    if model.originalFramework == 'keras' and framework == 'keras':
        error = "Exporting a existing keras model to a Keras file isn't \
supported yet"
        raise NotImplementedError(error)

    if model.originalFramework == 'keras' and framework == 'torch':
        error = "Conversions from keras to pytorch aren't supported yet"
        raise NotImplementedError(error)

    if model.originalFramework == 'torch' and framework == 'torch':
        error = "Exporting a existing pyTorch model to a pyTorch file isn't \
supported yet"
        raise NotImplementedError(error)

    return convert_torch2keras_file(model)


def convert(model, input_size=None, weights=True, quiet=True):
    """
    Converts a models between PyTorch and Keras

    Arguments:
        -model:
            the model to convert
        -input_size:
            int, list, or tuple.
            Optionnal if the model is very simple
        -weights (bool):
            Automatically convert weights
        -quiet (bool):
            If a progress bar should appear

    Returns:
        the model (LayerRepresentation)

    Raises:
        -NotImplementedError:
            if the model isn't supported yet
        -ValueError:
            if trying to convert from PyTorch to Keras without specifying
            input shape
    """

    if model.originalFramework == 'keras':
        error = "Conversions from keras to pytorch aren't supported yet"
        raise NotImplementedError(error)

    if model.originalFramework == 'torch':
        if input_size is None:
            raise ValueError("input_size is necessary to convert a model")

        input_size = normalizeShape(input_size)

        return convert_torch2keras(model,
                                   input_size=input_size,
                                   weights=weights,
                                   quiet=quiet)
