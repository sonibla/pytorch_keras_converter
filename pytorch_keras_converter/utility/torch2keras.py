from . import converting_layers as c_l
from tqdm import tqdm
from .LayerRepresentation import normalizeShape

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


lastProgress = 0


def kerasShape(tensor):
    """
    Determine the shape of a tensor or a keras layer

    Useful to check that PyTorch to Keras conversion doesn't fail
    """

    if tensor is None:
        return None
    else:
        if '_keras_shape' in dir(tensor):
            if tensor._keras_shape is not None:
                shape = tensor._keras_shape

                # In LayerRepresentation, we leave out batch size :
                shape = list(shape)
                del shape[0]
                shape = tuple(shape)

        elif 'shape' in dir(tensor):
            shape = tensor.shape.as_list()
            del shape[0]
            shape = tuple(shape)

        elif '_shape_val' in dir(tensor):
            if tensor._shape_val is not None:
                kerasShape = tensor._shape_val

                # In LayerRepresentation, we leave out batch size, so we
                # start at 1 (not 0) :
                values = range(1, len(kerasShape._dims))
                shape = [kerasShape._dims[k]._value for k in values]

                shape = tuple(shape)
        else:
            shape = None

    shape = normalizeShape(shape)
    return shape


def convert_torch2keras_file(model, input_size=None):
    createSimpleEquivalences(model, file=True)
    return model


def convert_torch2keras(model, input_size, weights=True, quiet=True):
    """
    Converts a pytroch model to keras

    Arguments:
        -model:
            the model to convert (LayerRepresentation)
        -input_size:
            int, list, or tuple.
        -weights (bool):
            If weights should be converted too (may take a lot of time !)
        -quiet (bool):
            If a progress bar should appear

    Returns:
        the model (LayerRepresentation)
    """

    global lastProgress
    lastProgress = 0

    # Step 1 : Compute all input and output shapes and place it on our model
    # Convert input_size into tulpe
    input_size = normalizeShape(input_size)

    if not quiet:
        progressBar = tqdm(total=model.numberOfChildren() + 1, unit='layer')
        print("\nAnalysing model...")
    else:
        progressBar = None

    findAllInputShapes(model, input_size)

    # Step 2: convert every simple layer (i.e native layers, in most cases)
    if not quiet:
        print("\nComputing equivalents layer by layer...")

    createSimpleEquivalences(model,
                             weights=weights,
                             quiet=quiet,
                             progressBar=progressBar)

    # Let's check if our model is fully converted:
    if 'keras' in model.equivalent.keys():
        return model

    # Step 3: keras Fonctionnal API
    if not quiet:
        print("\nConnecting layers together with Keras Functionnal API...")

    while 'keras' not in model.equivalent.keys():
        advancedKerasEquivalence(model,
                                 quiet=quiet,
                                 progressBar=progressBar)

    # Done!

    if not quiet:
        progressBar.close()
        print("\nDone !")

    return model


def createSimpleEquivalences(model,
                             file=False,
                             weights=True,
                             quiet=True,
                             progressBar=None):
    """
    Computes equivalent of most simple layers (native pyTorch layers,
    nn.Sequential containing only native layers...)

    Arguments:
        -model:
            A LayerRepresentation object to use
        -file (bool):
            If we want to write the equivalent in a python file
        -weights (bool):
            Also convert weights
        -quiet:
            If a progress bar should appear
        -progressBar:
            If a progress bar was already created, put it were
    """
    # Create a progress bar if necessary
    if not quiet and progressBar is None:
        progressBar = tqdm(total=model.numberOfChildren() + 1, unit='layer')

    if 'torch' in model.equivalent.keys():  # torch equivalent available
        # CONVERSION: torch -> keras
        if not model.children:  # 1st case: no children
            if model.isTorchBuiltIn():
                kerasEq = None
                kerasEq = c_l.torch2kerasEquivalent(model, weights=weights)

                kerasEqTxt = None
                if file:
                    kerasEqTxt = c_l.torch2kerasEquivalent(model,
                                                           file=True,
                                                           weights=weights)

            if kerasEq is not None:
                # keras equivalent computation succeeded!
                model.equivalent['keras'] = kerasEq

            if kerasEqTxt is not None:
                # keras equivalent computation succeeded!
                model.equivalentTxt['keras'] = kerasEqTxt

            if not quiet:
                updateProgress(model, progressBar)

        else:  # 2nd case: there are children
            if not model.childrenEquivalentsCompleted('keras',
                                                      file=file):
                # Trere are children,
                # but all equivalents aren't computed yet
                for child in model.children:
                    createSimpleEquivalences(child,
                                             file=file,
                                             weights=weights,
                                             quiet=quiet,
                                             progressBar=progressBar)

            # Here, we have computed all simple layers
            # If possible, we can still find an equivalent
            # if model is a container (sequential for example)
            success = model.childrenEquivalentsCompleted('keras')
            if model.isTorchBuiltIn() and success:
                kerasEq = c_l.torch2kerasEquivalent(model, weights=weights)
                if kerasEq is not None:
                    model.equivalent['keras'] = kerasEq
            if file:
                successTxt = model.childrenEquivalentsCompleted('keras',
                                                                file=True)
                if model.isTorchBuiltIn() and successTxt:
                    kerasEqTxt = c_l.torch2kerasEquivalent(model,
                                                           file=True,
                                                           weights=weights)
                    model.equivalentTxt['keras'] = kerasEqTxt

            if not quiet:
                updateProgress(model, progressBar)


def findAllInputShapes(model, pyTorch_input_size):
    """
    Finds input and output shapes of every layer in a model only knowing main
    input shape

    Arguments:
        -model:
            A LayerRepresentation object of the model to analsye
        -pyTorch_input_size:
            input shape

    Raises:
        -RuntimeError:
            If provided input shape isn't compatible with the model
    """
    if torch is None:
        raise ImportError("Could not import torch. Conversion failed !")

    pyTorchModel = model.equivalent['torch']

    def register_hook(module):

        def hook(module, Input, Output):

            identifier = id(module)

            # Input shape
            inputShape = list(Input[0].size())
            del inputShape[0]

            # Output shape
            if isinstance(Output, (list, tuple)):
                outputShape = [
                    list(o.size())[1:] for o in Output
                ]
            else:
                outputShape = list(Output.size())
            del outputShape[0]

            inputShape = normalizeShape(inputShape)
            outputShape = normalizeShape(outputShape)

            # Saving shapes
            selectedModel = model.getChildId(identifier, framework='torch')
            selectedModel.input_shape = inputShape
            selectedModel.output_shape = outputShape

        module.register_forward_hook(hook)

    # multiple inputs to the network
    if isinstance(pyTorch_input_size, tuple):
        pyTorch_input_size = [pyTorch_input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(1, *in_size) for in_size in pyTorch_input_size]

    # register hook
    pyTorchModel.apply(register_hook)

    # make a forward pass
    try:
        pyTorchModel(*x)
    except RuntimeError as err:
        raise RuntimeError('Failed to analyse pyTorch model !\n{}'.format(err))


def advancedKerasEquivalence(model,
                             quiet=True,
                             progressBar=None):
    """
    Uses keras Functionnal API to find all remaining equivalents

    Arguments:
        -model:
            A LayerRepresentation object to complete, or a list of
            LayerRepresentation objects to complete
        -quiet:
            If a progress bar should appear
        -progressBar:
            If a progress bar was already created, put it were
    """

    # Create a progress bar if necessary
    if not quiet and progressBar is None:
        progressBar = tqdm(total=model.numberOfChildren() + 1, unit='layer')

    if isinstance(model, list):
        # If we have to deal with a list of models:
        for oneModel in model:
            advancedKerasEquivalence(oneModel,
                                     quiet=quiet,
                                     progressBar=progressBar)
    else:
        if not quiet:
            updateProgress(model, progressBar)

        notKerasEquivExist = not('keras' in model.equivalent.keys())
        kerasOutputExist = model.kerasOutput is not None

        if notKerasEquivExist and model.childrenEquivalentsCompleted('keras'):
            c_l.spreadSignal(model)
            kerasOutputExist = model.kerasOutput is not None

            if kerasOutputExist:
                if model.name is not None:
                    kerasEq = keras.models.Model(inputs=model.kerasInput,
                                                 outputs=model.kerasOutput,
                                                 name=model.name)
                else:
                    kerasEq = keras.models.Model(inputs=model.kerasInput,
                                                 outputs=model.kerasOutput)
                model.equivalent['keras'] = kerasEq

        # Do the same to sub-sub-layers
        if not quiet:
            updateProgress(model, progressBar)
        advancedKerasEquivalence(model.children,
                                 quiet=quiet,
                                 progressBar=progressBar)


def updateProgress(model, progressBar):
    """
    During a conversion, updates the progress bar.
    Value is aucomatically computed using numberOfEquivalents

    Arguments:
        -model:
            A LayerRepresentation objest of one layer in the model being
            converted
        -progressBar:
            A ProgressBar object : the bar to update
    """
    global lastProgress
    mainParent = model.firstParent()
    progress = mainParent.numberOfEquivalents(framework='keras')
    diff = progress-lastProgress
    progressBar.update(diff)
    lastProgress = progress
