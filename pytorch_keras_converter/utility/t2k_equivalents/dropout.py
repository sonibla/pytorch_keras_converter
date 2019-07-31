try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def Dropout(model, file=False):
    """
    Converts a torch.nn.Dropout layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer Dropout to convert
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
    name = model.completeName()

    argumentsDropout = {'rate': pytorchLayer.p,
                        'input_shape': model.input_shape,
                        'name': name}

    argumentsScale = {'function': lambda x: x*(1/(1-pytorchLayer.p))}

    argumentsSequential = {}

    if not file:
        dropoutLayer = keras.layers.Dropout(**argumentsDropout)
        scaleLayer = keras.layers.Lambda(**argumentsScale)

        kerasLayer = keras.Sequential(**argumentsSequential)
        kerasLayer.add(dropoutLayer)
        kerasLayer.add(scaleLayer)
        return kerasLayer
    else:

        outstrDropout = 'keras.layers.Dropout('
        for arg, val in argumentsDropout.items():
            outstrDropout = outstrDropout + arg + '=' + str(val) + ', '
        outstrDropout = outstrDropout[:-2] + ')'

        outstrScale = 'keras.layers.Lambda(fonction= lambda x: x*(1/(1-'
        outstrScale = outstrScale + str(pytorchLayer.p) + ')))'

        outstr = 'keras.Sequential([\n    ' + outstrDropout + ',\n    '
        outstr = outstr + outstrScale + '\n], '

        for arg, val in argumentsSequential.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'

        return outstr
