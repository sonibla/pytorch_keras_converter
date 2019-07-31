"""
Functions used when saving a model after a conversion
"""
import os
from time import gmtime, strftime
import platform


def formatFilename(filename=None,
                   directory=None,
                   useTime=None):
    """
    Given filename and directory, formatFilename computes the file to actually
    create: if a file with this name already exist it wil generate another name
    by adding '_(0)' to filename.

    Arguments:
        -filename:
            Optionnal. The best name to give to the file. The program will
            search for an available name as similar as possible of filename
        -directory:
            Optionnal. Where to store the file
        -useTime (bool):
            Use current time in the filename. If set to None and no filename
            provided, this will be used

    Returns:
        An available filename with his directory (directory/filename) (or
        directory\filename if using windows)
    """

    def removeExtension(name):
        """
        Function that removes extension in a filename

        Argument:
            -name (str):
                The filename to analyse

        Returns:
            name, without the extension

            For example, if name == 'abc.def.ghi',
            removeExtension(name) == 'abc.def'
        """
        if '.' in name:
            return '.'.join(name.split('.')[:-1])

    def getExtension(name):
        """
        Function that returns the extension of a file

        Argument:
            -name (str):
                The filename to analyse

        Returns:
            The extension of name

            For example, if name == 'abc.def.ghi',
            getExtension(name) == '.ghi'
        """
        if '.' in name:
            return '.' + name.split('.')[-1]
        return ''

    # Use current time in filename if necessary:
    if filename is None:
        if useTime or useTime is None:
            currentTime = strftime("%d_%b_%Y_%H_%M_%S", gmtime())
            file = str(currentTime)
        else:
            file = ''
    else:
        file = str(filename)
        if useTime:
            currentTime = strftime("%d_%b_%Y_%H_%M_%S", gmtime())
            file = file + '_' + str(currentTime)

    # Remove border spaces:
    while file[-1] == ' ':
        file = file[:-1]
    while file[0] == ' ':
        file = file[1:]

    # Check that directory ends with '/' or '\' if using Windows
    if isinstance(directory, str):
        if platform.system() == 'Windows':
            if not(directory[-1] == '\\'):
                directory = directory + '\\'
        else:
            if not(directory[-1] == '/'):
                directory = directory + '/'
        file = directory + file

    # Add a number to find an available name
    if os.path.isfile(file):
        number = 0
        numStr = '_(' + str(number) + ')'
        newFile = removeExtension(file) + numStr + getExtension(file)
        while os.path.isfile(newFile):
            number += 1
            numStr = '_(' + str(number) + ')'
            newFile = removeExtension(file) + numStr + getExtension(file)
        file = newFile

    return file


def freeze(model):
    """
    Function that freezes a Keras model (inplace)

    Useful (sometimes necessary) before saving it in hdf5 format

    Argument:
        -model:
            A Keras model or layer
    """
    if 'layers' in dir(model):
        for layer in model.layers:
            layer.trainable = False
            if 'layers' in dir(layer):
                freeze(layer)
