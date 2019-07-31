try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None

import math


def MaxPool2d(model, file=False):
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
        -NotImplementedError:
            If dilation factor isn't 1

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
    kernel_size = pytorchLayer.kernel_size
    stride = pytorchLayer.stride
    dilation = pytorchLayer.dilation
    padding = pytorchLayer.padding
    ceil_mode = pytorchLayer.ceil_mode

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

    if not dilation == (1, 1):
        raise NotImplementedError("Error when converting MaxPool2d because \
dilation != 1 is not supported yet")

    # Take care of ceil_mode when computing output shape
    # Those formulas come from PyTorch documentation
    # https://pytorch.org/docs/stable/nn.html#maxpool2d
    Hin = model.input_shape[-2]
    Hout = (Hin+2*padding[0][0]-dilation[0]*(kernel_size[0]-1)-1/stride[0])+1

    Win = model.input_shape[-1]
    Wout = (Win+2*padding[1][0]-dilation[1]*(kernel_size[1]-1)-1/stride[1])+1

    # If ceil_mode is True, we may add a padding
    if ceil_mode and math.ceil(Hout) != math.floor(Hout):
        padding = ((padding[0][0], padding[0][1]+1),
                   (padding[1][0], padding[1][1]))

    if ceil_mode and math.ceil(Wout) != math.floor(Wout):
        padding = ((padding[0][0], padding[0][1]),
                   (padding[1][0], padding[1][1]+1))

    argumentsMaxpool = {'pool_size': kernel_size,
                        'strides': stride,
                        'padding': 'valid',
                        'data_format': 'channels_first',
                        'name': name}

    if padding == ((0, 0), (0, 0)):
        argumentsMaxpool['input_shape'] = model.input_shape

    argumentsPadding = {'padding': padding,
                        'input_shape': model.input_shape,
                        'data_format': 'channels_first'}

    argumentsSequential = {}

    if not file:
        maxpoolLayer = keras.layers.MaxPooling2D(**argumentsMaxpool)
        if padding == ((0, 0), (0, 0)):
            # No need to use a padding layer
            return maxpoolLayer

        paddingLayer = keras.layers.ZeroPadding2D(**argumentsPadding)
        kerasLayer = keras.Sequential(**argumentsSequential)
        kerasLayer.add(paddingLayer)
        kerasLayer.add(maxpoolLayer)

        return kerasLayer
    else:
        outstrMaxpool = 'keras.layers.MaxPooling2D('
        for arg, val in argumentsMaxpool.items():
            outstrMaxpool = outstrMaxpool + arg + '=' + str(val) + ', '
        outstrMaxpool = outstrMaxpool[:-2] + ')'

        outstrPadding = 'keras.layers.ZeroPadding2D('
        for arg, val in argumentsPadding.items():
            outstrPadding = outstrPadding + arg + '=' + str(val) + ', '
        outstrPadding = outstrPadding[:-2] + ')'

        outstr = 'keras.Sequential([\n    ' + outstrPadding + ',\n    '
        outstr = outstr + outstrMaxpool + '\n], '

        for arg, val in argumentsSequential.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'

        return outstr


def AvgPool2d(model, file=False):
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
        -NotImplementedError:
            if count_include_pad is False

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
    kernel_size = pytorchLayer.kernel_size
    stride = pytorchLayer.stride
    padding = pytorchLayer.padding
    ceil_mode = pytorchLayer.ceil_mode
    count_include_pad = pytorchLayer.count_include_pad

    if not count_include_pad:
        raise NotImplementedError("Error when converting AvgPool2d because \
count_include_pad == False is not supported \
yet")

    # Formatting them as tuple (height, width)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    # Formatting padding the Keras way ((top, bottom), (left, right))
    padding = ((padding[0], padding[0]),
               (padding[1], padding[1]))

    # Take care of ceil_mode when computing output shape
    # Those formulas come from PyTorch documentation
    # https://pytorch.org/docs/stable/nn.html#avgpool2d
    Hin = model.input_shape[-2]
    Hout = (Hin + 2*padding[0][0]-kernel_size[0] / stride[0]) + 1

    Win = model.input_shape[-1]
    Wout = (Win + 2*padding[1][0]-kernel_size[1] / stride[1]) + 1

    # If ceil_mode is True, we may add a padding
    if ceil_mode and math.ceil(Hout) != math.floor(Hout):
        padding = ((padding[0][0], padding[0][1]+1),
                   (padding[1][0], padding[1][1]))

    if ceil_mode and math.ceil(Wout) != math.floor(Wout):
        padding = ((padding[0][0], padding[0][1]),
                   (padding[1][0], padding[1][1]+1))

    argumentsAvgpool = {'pool_size': kernel_size,
                        'strides': stride,
                        'padding': 'valid',
                        'data_format': 'channels_first',
                        'name': name}

    if padding == ((0, 0), (0, 0)):
        argumentsAvgpool['input_shape'] = model.input_shape

    argumentsPadding = {'padding': padding,
                        'input_shape': model.input_shape,
                        'data_format': 'channels_first'}

    argumentsSequential = {}

    if not file:
        avgPoolLayer = keras.layers.AveragePooling2D(**argumentsAvgpool)
        if padding == ((0, 0), (0, 0)):
            # No need to use a padding layer
            return avgPoolLayer

        paddingLayer = keras.layers.ZeroPadding2D(**argumentsPadding)

        kerasLayer = keras.Sequential(**argumentsSequential)
        kerasLayer.add(paddingLayer)
        kerasLayer.add(avgPoolLayer)

        return kerasLayer
    else:
        outstrAvgpool = 'keras.layers.AveragePooling2D('
        for arg, val in argumentsAvgpool.items():
            outstrAvgpool = outstrAvgpool + arg + '=' + str(val) + ', '
        outstrAvgpool = outstrAvgpool[:-2] + ')'

        outstrPadding = 'keras.layers.ZeroPadding2D('
        for arg, val in argumentsPadding.items():
            outstrPadding = outstrPadding + arg + '=' + str(val) + ', '
        outstrPadding = outstrPadding[:-2] + ')'

        outstr = 'keras.Sequential([\n    ' + outstrPadding + ',\n    '
        outstr = outstr + outstrAvgpool + '\n], '

        for arg, val in argumentsSequential.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'

        return outstr


def AdaptiveAvgPool2d(model, file=False):
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
        -NotImplementedError:
            if output size isn't 1

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
    output_size = pytorchLayer.output_size

    if output_size == (1, 1):
        output_size = 1

    if output_size != 1:
        raise NotImplementedError("Error when converting AdaptiveAvgPool2d \
because output_size != 1 is not supported yet")

    argumentsAvgpool = {'data_format': 'channels_first',
                        'input_shape': model.input_shape,
                        'name': name}

    # Need a reshape so output shape is (channels, 1, 1) instead of (channels,)
    argumentsReshape = {'target_shape': model.output_shape}

    argumentsSequential = {}

    if not file:
        avgPoolLayer = keras.layers.GlobalAveragePooling2D(**argumentsAvgpool)
        reshapeLayer = keras.layers.Reshape(**argumentsReshape)

        kerasLayer = keras.Sequential(**argumentsSequential)
        kerasLayer.add(avgPoolLayer)
        kerasLayer.add(reshapeLayer)

        return kerasLayer
    else:
        raise NotImplementedError
        outstrAvgpool = 'keras.layers.GlobalAveragePooling2D('
        for arg, val in argumentsAvgpool.items():
            outstrAvgpool = outstrAvgpool + arg + '=' + str(val) + ', '
        outstrAvgpool = outstrAvgpool[:-2] + ')'

        outstrReshape = 'keras.layers.Lambda('
        for arg, val in argumentsReshape.items():
            outstrReshape = outstrReshape + arg + '=' + str(val) + ', '
        outstrReshape = outstrReshape[:-2] + ')'

        outstr = 'keras.Sequential([\n    ' + outstrAvgpool + ',\n    '
        outstr = outstr + outstrReshape + '\n], '

        for arg, val in argumentsSequential.items():
            outstr = outstr + arg + '=' + str(val) + ', '
        outstr = outstr[:-2] + ')'

        return outstr
