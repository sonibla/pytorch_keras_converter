"""
Compare 2 models on random data.

Available comparisons:
    - Keras Keras
    - Keras PyTorch
    - PyTorch PyTorch
    - PyTorch Keras

Useful to check a converted model's reliability
"""

try:
    import tensorflow.keras as keras
    import tensorflow
except ImportError:
    tensorflow = None
    try:
        import keras
    except ImportError:
        keras = None

# Same for pyTorch:
try:
    import torch
except ImportError:
    torch = None

import numpy as np
from .utility.LayerRepresentation import UsedFramework as UsedFrk
from .utility.LayerRepresentation import normalizeShape
from tqdm import tqdm


def pyTorchForwardPass(model, torchInput):
    """
    Make one forward pass on a PyTorch model (batchsize ==1), remove
    batchsize dimension, and convert it to a numpy array.

    Arguments:
        -model:
            A PyTorch model
        -torchInput:
            A PyTorch Tensor

    Returns:
        A numpy array of raw output data
    """
    out = model(torchInput).detach().numpy()
    out = np.squeeze(out, axis=0)
    return out


def kerasForwardPass(model, kerasInput):
    """
    Make one forward pass on a Keras model (batchsize ==1)

    Arguments:
        -model:
            A Keras model
        -torchInput:
            A Tensorflow Tensor

    Returns:
        A numpy array of raw output data
    """
    out = model.predict(kerasInput, steps=1)
    return out


def forwardPass(model, numpyInput):
    """
    Make one forward pass on a Keras or PyTorch model (batchsize ==1)

    Arguments:
        -model:
            A Keras or PyTorch model
        -numpyInput:
            A numPy array to feed the model

    Returns:
        A numpy array of raw output data

    Raises:
        -NotImplementedError:
            If provided model isn't supported
    """
    # Convert numpyInput to PyTorch Tensor:
    torchInput = torch.from_numpy(numpyInput)
    torchInput = torchInput.type(torch.FloatTensor)

    # Convert numpyInput to Tensorflow Tensor:
    if tensorflow is None:
        # Using Keras after making import keras
        kerasInput = keras.backend.tf.convert_to_tensor(torchInput.numpy())
    else:
        # Using Keras after making import tensorflow.keras as keras
        kerasInput = tensorflow.convert_to_tensor(torchInput.numpy())

    if UsedFrk(model) == 'torch':
        out = pyTorchForwardPass(model, torchInput)
    elif UsedFrk(model) == 'keras':
        out = kerasForwardPass(model, kerasInput)
    else:
        error = "Model {} not recognized".format(str(model))
        raise NotImplementedError(error)
    return out


def one_random_test(model1,
                    model2,
                    input_shape=None,
                    numpyInput=None,
                    input_range=None):
    """
    This function does one comparison between model1 and model2.
    If numpyInput is set to None, model1 and model2 will be testd on random
    data

    Arguments:
        -model1:
            A PyTorch or Keras model
        -model2:
            A PyTorch or Keras model
        -input_shape:
            A list, int or tuple of the input shape (without batchsize)
            Optionnal if numpyInput provided
        -numpyInput:
            A numpy array containing the data to test the models on.
            Optionnal of input_shape provided
        -input_range:
            Optionnal.
            A list of 2 elements containing max and min values to give as
            input to the model. If None, models will be tested on
            samples from the "standard normal" distribution.

    Returns:
        A tuple of 2 numpy arrays containing raw output data of model1 and
        model2

    """
    if torch is None or keras is None:
        raise ImportError("pyTorch or Keras unavailable!")

    if numpyInput is None:
        # Generate random data:

        # First, normalize input_shape as a tuple:
        input_shape = normalizeShape(input_shape)

        if input_range is None:
            # Generate random data from the "standard normal" distribution:
            numpyInput = np.random.randn(1, *input_shape)
        else:
            # Generate uniform random data between min(input_range) and
            # max(input_range):
            r = input_range
            randArray = np.random.rand(1, *input_shape)
            numpyInput = (randArray * (max(r)-min(r))) + min(r)

    # Make a forward pass for each model:
    out1 = forwardPass(model1, numpyInput)
    out2 = forwardPass(model2, numpyInput)

    return out1, out2


def many_random_tests(model1,
                      model2,
                      input_shape,
                      number=100,
                      input_range=None):
    """
    This function does many comparisons between model1 and model2 on *one*
    random Tensor

    Arguments:
        -model1:
            A PyTorch or Keras model
        -model2:
            A PyTorch or Keras model
        -input_shape:
            A list, int or tuple of the input shape (without batchsize)
        -number:
            Number of tests to perform on models. It should be set to 1 if
            models don't have any random behavior, such as Drouput layers
        -input_range:
            Optionnal.
            A list of 2 elements containing max and min values to give as
            input to the model. If None, models will be tested on
            samples from the "standard normal" distribution.

    Returns:
        A tuple of 2 numpy arrays containing means of output data from model1
        and model2

    """
    if torch is None or keras is None:
        raise ImportError("pyTorch or Keras unavailable!")

    # First, normalize input_shape as a tuple:
    input_shape = normalizeShape(input_shape)

    # Lists to store raw output data
    testsModel1 = list()
    testsModel2 = list()

    if input_range is None:
        # Generate random data from the "standard normal" distribution:
        numpyInput = np.random.randn(1, *input_shape)
    else:
        # Generate uniform random data between min(input_range) and
        # max(input_range):
        r = input_range
        randArray = np.random.rand(1, *input_shape)
        numpyInput = (randArray * (max(r)-min(r))) + min(r)

    for _ in range(number):
        # Perform several tests on the same input array
        out1, out2 = one_random_test(model1,
                                     model2,
                                     input_shape,
                                     numpyInput,
                                     input_range=input_range)
        testsModel1.append(out1)
        testsModel2.append(out2)

    mean1 = np.mean(testsModel1, axis=0)
    mean2 = np.mean(testsModel2, axis=0)

    return mean1, mean2


def standard_test(model1,
                  model2,
                  input_shape,
                  input_range=None,
                  numberA=10,
                  numberB=2,
                  quiet=False):
    """
    This function does many comparisons between model1 and model2 on *several*
    random Tensors

    The more numberA and numberB are high, the more accurate the test will be,
    but it will become very slower.
    In total, we have to make numberA*numberB*2 forward passes !

    Arguments:
        -model1:
            A PyTorch or Keras model
        -model2:
            A PyTorch or Keras model
        -input_shape:
            A list, int or tuple of the input shape (without batchsize)
        -input_range:
            Optionnal.
            A list of 2 elements containing max and min values to give as
            input to the model. If None, models will be tested on
            samples from the "standard normal" distribution.
        -numberA:
            Number of tests to perform on models (i.e number of random input
            tensor to generate and test the models on)
        -numberB:
            Number of tests to perform on models using each random tensor.
            It should be set to 1 if models don't have any random behavior,
            such as Drouput layers
        -quiet (bool):
            If a progress bar should appear

    Returns:
        A numpy array containing the differences between models for each output

    """

    if not quiet:
        bar = tqdm(total=numberA)

    # First, normalize input_shape as a tuple:
    input_shape = normalizeShape(input_shape)

    differences = list()

    for _ in range(numberA):
        if not quiet:
            bar.update()

        # Perform numberA tests. Each test will generate a random tensor and
        # make numberB forward passes on each model
        # In total, we have to make numberA*numberB*2 forward passes
        out1, out2 = many_random_tests(model1,
                                       model2,
                                       input_shape,
                                       number=numberB,
                                       input_range=input_range)

        # Reshape out1 and out2 in a 1-dimension array. This is not necessary
        out1 = np.reshape(out1, -1)
        out2 = np.reshape(out2, -1)

        diff = out1 - out2
        differences.append(diff)

    if not quiet:
        bar.close()

    return np.array(differences)


def comparison(model1, model2, input_shape, input_range=None, quiet=True):
    """
    This function does a complete comparison between model1 and model2.

    Arguments:
        -model1:
            A PyTorch or Keras model
        -model2:
            A PyTorch or Keras model
        -input_shape:
            A list, int or tuple of the input shape (without batchsize)
        -input_range:
            Optionnal.
            A list of 2 elements containing max and min values to give as
            input to the model. If None, models will be tested on
            samples from the "standard normal" distribution.
        -quiet (bool):
            If a progress bar should appear

    Returns:
        The mean squared error between the two models
        If models are identical, it should be below 1e-10

    """

    if model1 == model2 and not quiet:
        print("Those models are identical twins...")
        if model1 is model2:
            print("Actually they are the same python object.")
        print("\nWe should have a really, really low MSE.")
        print("Use Ctrl+C to cnacel")

    # First, normalize input_shape as a tuple:
    input_shape = normalizeShape(input_shape)

    differences = standard_test(model1,
                                model2,
                                input_shape,
                                input_range=input_range,
                                quiet=quiet)

    MSE = float(np.mean(np.square(np.reshape(differences, -1))))

    if not quiet:
        print("\nMean Squared Error: MSE={}".format(round(MSE, 3)))

    return MSE
