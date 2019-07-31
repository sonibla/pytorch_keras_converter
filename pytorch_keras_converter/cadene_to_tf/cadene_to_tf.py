try:
    import pretrainedmodels
except ImportError:
    print("\nWarning: Cadene's pretrainedmodels unavailable !")
    print("Try: pip install pretrainedmodels")

from .. import API
from .. import supported_cadene_models as supported
from . import utils


def sync_model(model_name,
               model_params=None,
               outputDirectory=None,
               quiet=True):
    """
    sync_model : automatically converts and save one Cadene model

    Arguments:
        -model_name:
            The model's name
        -model_params (dict):
            Optionnal
            If you want a particular config for the model
            Example:
            {'pretrained': None}
        -outputDirectory (str):
            Optionnal. Where .h5 files should be saved
        -quiet (bool):
            If False, display a progress bar

    Returns:
        The created file
    """

    if model_name not in pretrainedmodels.model_names:
        print("Warning: model {} not recognized ! Skipping".format(model_name))
    elif model_name not in supported:
        print("Warning: model {} not supported ! Skipping".format(model_name))
    else:
        if not quiet:
            print("Converting {}\n".format(model_name))
        name = model_name

        if model_params is None or model_params == dict():
            model = getattr(pretrainedmodels, model_name)()
        else:
            model = getattr(pretrainedmodels, model_name)(**model_params)
            name = name + '('
            for param, value in model_params.items():
                name = name + param + '=' + str(value) + ', '
            name = name[:-2] + ')'

        settings = pretrainedmodels.pretrained_settings[model_name]

        if 'pretrained' in model_params.keys() and \
           model_params['pretrained'] is not None:

            pretrained = model_params['pretrained']
            input_shape = settings[pretrained]['input_size']
            input_range = settings[pretrained]['input_range']
        else:
            for pretrained in settings.keys():
                input_shape = settings[pretrained]['input_size']
                input_range = settings[pretrained]['input_range']
                break

        try:
            file = API.convert_and_save(model,
                                        input_shape=input_shape,
                                        weights=True,
                                        quiet=quiet,
                                        ignore_tests=False,
                                        input_range=input_range,
                                        filename=name,
                                        directory=outputDirectory)
            return file
        except RuntimeWarning:
            if not quiet:
                print("Failed to convert model {}".format(model_name))
            return None


def cadene_to_tf(modelList=None, outputDirectory=None, quiet=True):
    """
    cadene_to_tf : automatically converts and save Cadene's models

    Arguments:
        -modelList:
            A tuple of names of the models to convert
            OR
            A list of names of the models to convert
            OR
            A str telling the emplacement of a file containing names
            of models to convert (one model per line)

            If you want a particular config for each model, put it between
            parenthesis after model's name, for example:

            'se_resnet50(pretrained=None)'
        -outputDirectory (str):
            Optionnal. Where .h5 files should be saved
        -quiet (bool):
            If False, display a progress bar

    Raises:
        -TypeError:
            If modelList wasn't a str, list, or tuple

    Returns:
        A list of created files

    Examples:
        cadene_to_tf(modelList=['se_resnet50'], quiet=False)

        cadene_to_tf(modelList=['se_resnet50',
                                'bninception(pretrained=None)])

        cadene_to_tf(modelList=models.txt)
    """

    if modelList is None:
        return None

    if isinstance(modelList, str):
        with open(modelList, 'r') as file:
            models = file.read().split('\n')
    elif isinstance(modelList, tuple):
        models = list(modelList)
    elif isinstance(modelList, list):
        models = modelList
    else:
        raise TypeError('modelList has to be str, list, or tuple')

    fileList = []
    failed = []

    for i in range(len(models)):
        assert isinstance(models[i], str)
        models[i] = utils.removeBorderSpaces(models[i])
        if '(' in models[i] or ')' in models[i]:
            extract = utils.extractFunctionArguments(models[i])
            model_name = utils.removeBorderSpaces(extract[0])
            model_params = extract[1]
        else:
            model_name = models[i]
            model_params = dict()
        if len(model_name) > 0:
            file = sync_model(model_name,
                              model_params,
                              outputDirectory=outputDirectory,
                              quiet=quiet)
            if file is not None:
                fileList.append(file)
            else:
                failed.append(model_name)

    if failed and not quiet:
        print("\nWarning: some models were not converted !")
        print(str(failed))

    return fileList
