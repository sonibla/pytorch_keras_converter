from . import t2k_equivalents as t2k

try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None
try:
    import torch
except ImportError:
    torch = None


def spreadSignal(model):
    """
    Converts Cadene models using Keras Functionnal API.

    This function creates inputs (keras.layers.Input) on first layers of each
    block, and 'connects' layers together

    A connection between layer0 and layer1 means that output of layer0 goes to
    input of layer1.

    Argument:
        -model:
            The LayerRepresentation of the layer or model to convert
    """

    if model.firstParent().type == 'InceptionV4':
        # Actually it will raise an exception because of:
        # nn.AvgPool2d(count_include_pad=False)
        t2k.cadene.InceptionV4.spreadSignal(model)

    elif model.firstParent().type == 'BNInception':
        t2k.cadene.BNInception.spreadSignal(model)

    elif model.firstParent().type == 'SENet':
        t2k.cadene.SENet.spreadSignal(model)

    elif model.firstParent().type == 'ResNet':
        t2k.cadene.ResNet.spreadSignal(model)

    elif model.firstParent().type == 'FBResNet':
        t2k.cadene.FBResNet.spreadSignal(model)

    elif model.type == 'Sequential':
        model.ConnectModelInputToChildren('0')
        for i in range(len(model.children)-1):
            model.Connect2Layers(str(i), str(i+1))
        model.ConnectChildrenOutputToModel(str(len(model.children)-1))

    else:
        err = "Warning: layer or model '{}' not recognized!".format(model.type)
        raise NotImplementedError(err)


def torch2kerasEquivalent(model, file=False, weights=True):
    """
    Converts a pytorch native layer or container into a keras layer
    All children must have their equivalents already computed

    Arguments:
        -model:
            A LayerRepresentation object of the layer to convert
        -file (bool):
            If we want to write the equivalent in a python file
        -weights (bool):
            Also convert weights

    Raises:
        -ImportError:
            If Keras isn't available
        -NotImplementedError:
            If the given layer isn't supported yet

    Returns:
        Keras equivalent.
        If file is True, returns as a str to put in a python file
        Else, return the keras layer
    """
    if keras is None:
        raise ImportError("Could not import keras. Conversion failed !")

    if model.detailedType == 'torch.nn.modules.container.Sequential':
        return t2k.Sequential(model, file=file)

    if model.detailedType == 'torch.nn.modules.conv.Conv2d':
        return t2k.Conv2d(model, file=file, weights=weights)

    if model.detailedType == 'torch.nn.modules.activation.ReLU':
        return t2k.ReLU(model, file=file)

    if model.detailedType == 'torch.nn.modules.activation.Sigmoid':
        return t2k.Sigmoid(model, file=file)

    if model.detailedType == 'torch.nn.modules.batchnorm.BatchNorm2d':
        return t2k.BatchNorm2d(model, file=file, weights=weights)

    if model.detailedType == 'torch.nn.modules.dropout.Dropout':
        return t2k.Dropout(model, file=file)

    if model.detailedType == 'torch.nn.modules.linear.Linear':
        return t2k.Linear(model, file=file, weights=weights)

    if model.detailedType == 'torch.nn.modules.padding.ZeroPad2d':
        return t2k.ZeroPad2d(model, file=file)

    if model.detailedType == 'torch.nn.modules.pooling.AdaptiveAvgPool2d':
        return t2k.AdaptiveAvgPool2d(model, file=file)

    if model.detailedType == 'torch.nn.modules.pooling.MaxPool2d':
        return t2k.MaxPool2d(model, file=file)

    if model.detailedType == 'torch.nn.modules.pooling.AvgPool2d':
        return t2k.AvgPool2d(model, file=file)

    err = "Layers of type {} aren't implemented yet".format(model.detailedType)
    raise NotImplementedError(err)
