"""
Simple API to convert models between PyTorch and Keras

(Conversions from Keras to PyTorch aren't implemented)
"""
from . import utility
from . import tests
from . import io_utils as utils
import tensorflow


def convert(model,
            input_shape,
            weights=True,
            quiet=True,
            ignore_tests=False,
            input_range=None,
            save=None,
            filename=None,
            directory=None):
    """
    Conversion between PyTorch and Keras
    (Conversions from Keras to PyTorch aren't implemented)

    Arguments:
        -model:
            A Keras or PyTorch model or layer to convert
        -input_shape:
            Input shape (list, tuple or int), without batchsize.
        -weights (bool):
            Also convert weights. If set to false, only convert model
            architecture
        -quiet (bool):
            If a progress bar and some messages should appear
        -ignore_tests (bool):
            If tests should be ignored. If set to False, converted model will
            still be tested by security. If models are not identical, it will
            only print a warning.

            If set to True, and models are not identical, RuntimeWarning will
            be raised
        -input_range:
            Optionnal.
            A list of 2 elements containing max and min values to give as
            input to the model when performing the tests. If None, models will
            be tested on samples from the "standard normal" distribution.
        -save:
            If model should be exported to a hdf5 file.
        -filename:
            Filename to give to model's hdf5 file. If filename is not None and
            save is not False, then save will automatically be set to True
        -directory:
            Where to save model's hdf5 file. If directory is not None and
            save is not False, then save will automatically be set to True

    Raises:
        -RuntimeWarning:
            If converted and original model aren't identical, and ignore_tests
            is False

    Returns:
        If model has been exported to a file, it will return the name of the
        file

        Else, it returns the converted model
    """

    if (filename is not None or directory is not None) and save is None:
        save = True
    if save is None:
        save = False

    if not quiet:
        print('\nConversion...')

    # Converting:
    newModel = utility.convert(model=utility.LayerRepresentation(model),
                               input_size=input_shape,
                               weights=weights,
                               quiet=quiet)

    # Actually, newModel is a LayerRepresentation object
    # Equivalents:
    torchModel = newModel.equivalent['torch']
    kerasModel = newModel.equivalent['keras']

    if not quiet:
        print('Automatically testing converted model reliability...\n')

    # Checking converted model reliability
    tested = False
    try:
        meanSquaredError = tests.comparison(model1=torchModel,
                                            model2=kerasModel,
                                            input_shape=input_shape,
                                            input_range=input_range,
                                            quiet=quiet)
        tested = True
    except tensorflow.errors.InvalidArgumentError:
        print("Warning: tests unavailable!")

    if tested and meanSquaredError > 0.0001:
        if ignore_tests:
            print("Warning: converted and original models aren't identical !\
(mean squared error: {})".format(meanSquaredError))
        else:
            raise RuntimeWarning("Original and converted model do not match !\
\nOn random input data, outputs showed a mean squared error of {} (if should \
be below 1e-10)".format(meanSquaredError))
    elif not quiet and tested:
        print('\n Original and converted models match !\nMean squared err\
or : {}'.format(meanSquaredError))

    if save:
        if not quiet:
            print('Saving model...')

        defaultName = 'conversion_{}'.format(newModel.name)

        if filename is None:
            filename = defaultName

        # Formatting filename so that we don't overwrite any existing file
        file = utils.formatFilename(filename,
                                    directory)

        # Freezing Keras model (trainable = False everywhere)
        utils.freeze(kerasModel)

        # Save the entire model
        kerasModel.save(file + '.h5')

        if not quiet:
            print('Done !')

        return file + '.h5'

    if not quiet:
        print('Done !')

    return kerasModel


def convert_and_save(model,
                     input_shape,
                     weights=True,
                     quiet=True,
                     ignore_tests=False,
                     input_range=None,
                     filename=None,
                     directory=None):
    """
    Conversion between PyTorch and Keras, and automatic save
    (Conversions from Keras to PyTorch aren't implemented)

    Arguments:
        -model:
            A Keras or PyTorch model or layer to convert
        -input_shape:
            Input shape (list, tuple or int), without batchsize.
        -weights (bool):
            Also convert weights. If set to false, only convert model
            architecture
        -quiet (bool):
            If a progress bar and some messages should appear
        -ignore_tests (bool):
            If tests should be ignored. If set to False, converted model will
            still be tested by security. If models are not identical, it will
            only print a warning.

            If set to True, and models are not identical, RuntimeWarning will
            be raised
        -input_range:
            Optionnal.
            A list of 2 elements containing max and min values to give as
            input to the model when performing the tests. If None, models will
            be tested on samples from the "standard normal" distribution.
        -filename:
            Filename to give to model's hdf5 file. If filename is not None and
            save is not False, then save will automatically be set to True
        -directory:
            Where to save model's hdf5 file. If directory is not None and
            save is not False, then save will automatically be set to True

    Returns:
        Name of created hdf5 file
    """

    return convert(model=model,
                   input_shape=input_shape,
                   weights=weights,
                   quiet=quiet,
                   ignore_tests=ignore_tests,
                   input_range=input_range,
                   save=True,
                   filename=filename,
                   directory=directory)
