try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def BatchNorm2d(model, file=False, weights=True):
    """
    Converts a torch.nn.BatchNorm2d layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer BatchNorm2d to convert
        -file (bool):
            If we want to write the equivalent in a python file
        -weights (bool):
            Also convert weights

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

    argumentsBatchNorm = {'axis': 1,
                          'momentum': pytorchLayer.momentum,
                          'epsilon': pytorchLayer.eps,
                          'center': pytorchLayer.affine,
                          'scale': pytorchLayer.affine,
                          'input_shape': model.input_shape,
                          'name': name}

    if weights:
        parameters = dict()
        for key, val in dict(pytorchLayer.state_dict()).items():
            # Convert every parameter Tensor to a numpy array
            parameters[key] = val.detach().numpy()

        # List of [weight, bias, running_mean, running_var]
        paramList = [parameters['weight'],
                     parameters['bias'],
                     parameters['running_mean'],
                     parameters['running_var']]

    if not file:
        BatchNormLayer = keras.layers.BatchNormalization(**argumentsBatchNorm)

        kerasLayer = keras.Sequential()
        kerasLayer.add(BatchNormLayer)

        if weights:
            kerasLayer.layers[0].set_weights(paramList)

        return kerasLayer
    else:
        outstr = 'keras.layers.BatchNormalization('
        for arg, val in argumentsBatchNorm.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'
        return outstr
