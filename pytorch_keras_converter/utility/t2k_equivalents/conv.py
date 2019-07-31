try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def Conv2d(model, file=False, weights=True):
    """
    Converts a torch.nn.Conv2d layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer Conv2d to convert
        -file (bool):
            If we want to write the equivalent in a python file
        -weights (bool):
            Also convert weights

    Raises:
        -ImportError:
            If Keras import failed
        -RuntimeError:
            If shapes don't match
        -NotImplementedError:
            if groups != 1 or padding isn't zero-padding

    Returns:
        Keras equivalent.
        If file is True, returns as a str to put in a python file
        Else, return the keras layer
    """
    if keras is None:
        raise ImportError("Could not import keras. Conversion failed !")

    pytorchLayer = model.equivalent['torch']
    name = model.completeName()

    # Getting hyper parameters
    in_channels = pytorchLayer.in_channels
    out_channels = pytorchLayer.out_channels
    kernel_size = pytorchLayer.kernel_size
    stride = pytorchLayer.stride
    padding = pytorchLayer.padding
    dilation = pytorchLayer.dilation
    groups = pytorchLayer.groups
    padding_mode = pytorchLayer.padding_mode
    bias = 'bias' in dict(pytorchLayer.named_parameters()).keys()

    # A little verification
    if in_channels != model.input_shape[0]:
        raise RuntimeError("Error when converting Conv2d, shapes don't match")

    if groups != 1:
        raise NotImplementedError("Error when converting Conv2d because \
groups != 1 is not supported yet")

    if padding_mode != "zeros":
        raise NotImplementedError("Error when converting Conv2d because \
padding_mode != 'zeros' is not supported yet")

    # Formatting them as tuple (height, width)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)

    # Formatting padding the Keras way ((top, bottom), (left, right))
    padding = ((padding[0], padding[0]),
               (padding[1], padding[1]))

    argumentsConv = {'filters': out_channels,
                     'kernel_size': kernel_size,
                     'strides': stride,
                     'padding': 'valid',
                     'data_format': 'channels_first',
                     'dilation_rate': dilation,
                     'use_bias': bias,
                     'name': name}

    argumentsPadding = {'padding': padding,
                        'data_format': 'channels_first',
                        'input_shape': model.input_shape}

    if padding == ((0, 0), (0, 0)):
        argumentsConv['input_shape'] = model.input_shape

    if weights:
        parametersConv = dict()
        for key, val in dict(pytorchLayer.state_dict()).items():
            # Convert every parameter Tensor to a numpy array
            parametersConv[key] = val.detach().numpy()
            if key == 'weight':
                # Weights array also need to be transposed
                parametersConv[key] = parametersConv[key].transpose(2, 3, 1, 0)

        # List of [weight, bias]
        paramListConv = [parametersConv['weight']]
        if 'bias' in parametersConv.keys():
            paramListConv.append(parametersConv['bias'])

    if not file:
        convLayer = keras.layers.Conv2D(**argumentsConv)

        paddingLayer = keras.layers.ZeroPadding2D(**argumentsPadding)

        kerasLayer = keras.Sequential()
        if not padding == ((0, 0), (0, 0)):
            kerasLayer.add(paddingLayer)
        kerasLayer.add(convLayer)

        if weights:
            kerasLayer.layers[-1].set_weights(paramListConv)

        return kerasLayer
    else:
        outstrConv = 'keras.layers.Conv2D('
        for arg, val in argumentsConv.items():
            outstrConv = outstrConv + arg + '=' + str(val) + ', '
        outstrConv = outstrConv[:-2] + ')'

        outstrPadding = 'keras.layers.ZeroPadding2D('
        for arg, val in argumentsPadding.items():
            outstrPadding = outstrPadding + arg + '=' + str(val) + ', '
        outstrPadding = outstrPadding[:-2] + ')'

        outstr = 'keras.Sequential([\n    ' + outstrPadding + ',\n    '
        outstr = outstr + outstrConv + '\n])'

        return outstr
