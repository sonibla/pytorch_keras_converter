try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def Linear(model, file=False, weights=True):
    """
    Converts a torch.nn.Linear layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer Linear to convert
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

    bias = 'bias' in dict(pytorchLayer.named_parameters()).keys()

    argumentsDense = {'units': pytorchLayer.out_features,
                      'use_bias': bias,
                      'input_shape': model.input_shape,
                      'name': name}

    if weights:
        parametersDense = dict()
        for key, val in dict(pytorchLayer.state_dict()).items():
            # Convert every parameter Tensor to a numpy array
            parametersDense[key] = val.detach().numpy()
            if key == 'weight':
                # Weights array also need to be transposed
                parametersDense[key] = parametersDense[key].transpose(1, 0)

        # List of [weight, bias]
        paramList = [parametersDense['weight']]
        if 'bias' in parametersDense.keys():
            paramList.append(parametersDense['bias'])

    if not file:
        DenseLayer = keras.layers.Dense(**argumentsDense)

        kerasLayer = keras.Sequential()
        kerasLayer.add(DenseLayer)

        if weights:
            kerasLayer.layers[0].set_weights(paramList)

        return kerasLayer
    else:
        outstr = 'keras.layers.Dense('
        for arg, val in argumentsDense.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'
        return outstr
