try:
    import tensorflow.keras as keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


def Sequential(model, file=False):
    """
    Converts a torch.nn.Sequential layer

    Arguments:
        -model:
            A LayerRepresentation object of the layer Sequential to convert
        -file (bool):
            If we want to write the equivalent in a python file

    Raises:
        -ImportError:
            If Keras import failed

    Returns:
        Keras equivalent.
        If file is True, returns as a str to put in a python file
        Else, return the keras layer

        If layers don't have equivaents yet, returns None
    """
    if keras is None:
        raise ImportError("Could not import keras. Conversion failed !")

    name = model.completeName()

    if not file:
        kerasLayer = keras.Sequential(name=name)

        lNumber = -1
        # First, we need to sort layers
        subLayersDict = dict()
        for child in model.children:
            if 'keras' not in child.equivalent.keys():
                return None
            try:
                # If layers aren't named,
                # PyTorch uses default named '0', '1', '2',...
                lNumber = int(child.name)
            except ValueError:
                lNumber += 1
            subLayersDict[lNumber] = child.equivalent['keras']

        subLayersList = [None]*subLayersDict.__len__()

        for number, subLayer in subLayersDict.items():
            subLayersList[number] = subLayer

        if None in subLayersList:
            return None

        for subLayer in subLayersList:
            kerasLayer.add(subLayer)

        return kerasLayer
    else:
        lNumber = -1
        # First, we need to sort layers
        subLayersDict = dict()
        for child in model.children:
            if 'keras' not in child.equivalentTxt.keys():
                return None
            try:
                # If layers aren't named,
                # PyTorch uses default named '0', '1', '2',...
                lNumber = int(child.name)
            except ValueError:
                lNumber += 1
            subLayersDict[lNumber] = child.equivalentTxt['keras']

        subLayersList = [None]*subLayersDict.__len__()

        for number, subLayerTxt in subLayersDict.items():
            subLayersList[number] = subLayerTxt

        if None in subLayersList:
            return None

        outstr = 'keras.Sequential(['

        for subLayerTxt in subLayersList:
            outstr = outstr + '\n    ' + subLayerTxt + ','

        outstr = outstr[:-1] + '\n], name=' + name + ')'

        return outstr
