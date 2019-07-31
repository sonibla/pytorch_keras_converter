try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def ReLU(model, file=False):
    """
    Converts a torch.nn.ReLU layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer ReLU to convert
        -file (bool):
            If we want to write the equivalent in a python file

    Raises:
        -ImportError:
            If Keras import failed

    Returns:
        Keras equivalent.
        If file is True, returns as a str to put in a python file
        Else, return the keras layer
    """
    if keras is None:
        raise ImportError("Could not import keras. Conversion failed !")

    name = model.completeName()

    arguments = {'input_shape': model.input_shape,
                 'max_value': None,
                 'negative_slope': 0.0,
                 'threshold': 0.0,
                 'name': name}

    if not file:
        kerasLayer = keras.layers.ReLU(**arguments)
        return kerasLayer
    else:
        outstr = 'keras.layers.ReLU('
        for arg, val in arguments.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'
        return outstr


def Sigmoid(model, file=False):
    """
    Converts a torch.nn.Sigmoid layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer Sigmoid to convert
        -file (bool):
            If we want to write the equivalent in a python file

    Raises:
        -ImportError:
            If Keras import failed

    Returns:
        Keras equivalent.
        If file is True, returns as a str to put in a python file
        Else, return the keras layer
    """
    if keras is None:
        raise ImportError("Could not import keras. Conversion failed !")

    name = model.completeName()

    arguments = {'activation': 'sigmoid',
                 'input_shape': model.input_shape,
                 'name': name}

    if not file:
        kerasLayer = keras.layers.Activation(**arguments)
        return kerasLayer
    else:
        outstr = 'keras.layers.Activation('
        for arg, val in arguments.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'
        return outstr
