try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def ZeroPad2d(model, file=False):
    """
    Converts a torch.nn.ZeroPad2d layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer ZeroPad2d to convert
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

    pytorchLayer = model.equivalent['torch']
    name = name = model.completeName()

    padding = pytorchLayer.padding

    if isinstance(padding, int):
        left = padding
        right = padding
        bottom = padding
        top = padding
    elif padding.__len__() == 2:
        left = padding[0]
        right = padding[1]
        top = 0
        bottom = 0
    elif padding.__len__() == 4:
        left = padding[0]
        right = padding[1]
        top = padding[2]
        bottom = padding[3]

    arguments = {'padding': ((top, bottom), (left, right)),
                 'data_format': 'channels_first',
                 'input_shape': model.input_shape,
                 'name': name}

    if not file:
        kerasLayer = keras.layers.ZeroPadding2D(**arguments)
        return kerasLayer
    else:
        outstr = 'keras.layers.ZeroPadding2D('
        for arg, val in arguments.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'
        return outstr
