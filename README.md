# pytorch_keras_converter

A PyTorch-Keras converter made for [Cadene](https://github.com/Cadene)'s [pretrained models](https://github.com/cadene/pretrained-models.pytorch).

Also converts some simple PyTorch models. See [supported layers](https://github.com/sonibla/pytorch_keras_converter#other-models) for more details.

## Installation

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

[Python](https://www.python.org/downloads/) : supported versions : >=3.5

You can also install python with [Anaconda](https://www.anaconda.com/distribution/#download-section).

### Installing

This command should install automatically `pytorch_keras_converter` and every dependency:
```
python3 setup.py install --user
```

To install on a particular version of Python (here 3.7):
```
python3.7 setup.py install --user
```

To install on the entire system (requires administrator privileges):
```
sudo python setup.py install
```

### Troubleshooting

#### Installing `pip` or `setuptools`

If modules `pip` or `setuptools` aren't installed on your Python environment:
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --user
```

#### Manually installing dependencies

Dependencies should install automatically when running `setup.py`. But if it fails, install them manually:
```
python -m pip install torch pretrainedmodels tensorflow graphviz numpy h5py tqdm --user
```

Remove `--user` to install on the whole system, replace `python` with `python3.7` to install on Python 3.7.

## Usage

### Quick examples

- To convert `se_resnet50`:
```
>>> import pytorch_keras_converter as pkc
>>> pkc.cadene_to_tf(['se_resnet50'], quiet=False)
```

- To convert `bninception` and `fbresnet152`:
```
>>> import pytorch_keras_converter as pkc
>>> pkc.cadene_to_tf(['bninception', 'fbresnet152'])
```

- To convert `se_resnet152` with random weights:
```
>>> import pytorch_keras_converter as pkc
>>> pkc.cadene_to_tf(['se_resnet152(pretrained=None)'])
```

- To automatically convert many models:

Create a file containing on each line one model's name. For example:
```
resnet18
resnet34
resnet50(pretrained=None)
resnet50
```

Let's store this file in `models.txt`.

Now, in a Python shell:
```
>>> import pytorch_keras_converter as pkc
>>> pkc.cadene_to_tf('models.txt')
```

### API documentation

#### `pytorch_keras_converter.API.convert`

```
def convert(model,
            input_shape,
            weights=True,
            quiet=True,
            ignore_tests=False,
            input_range=None,
            save=None,
            filename=None,
            directory=None):
```

##### Arguments:

- **model**:
A Keras or PyTorch model or layer to convert
- **input_shape** (list, tuple or int):
Input shape, without batchsize.
- **weights** (bool):
Also convert weights. If set to *False*, only convert model
architecture
- **quiet** (bool):
If *False*, display a progress bar and some messages
- **ignore_tests** (bool):
If tests should be ignored. 
  - If set to *False*, converted model will
still be tested by security. If models are not identical, it will
only print a warning.
  - If set to *True*, and models are not identical, *RuntimeWarning* will
be raised
- **input_range**:
Optional.
A list of 2 elements containing max and min values to give as
input to the model when performing the tests. If *None,* models will
be tested on samples from the "standard normal" distribution.
- **save**:
If model should be exported to a hdf5 file.
- **filename**:
Optional.
Filename to give to model's hdf5 file. If filename is not *None* and
save is not *False*, then save will automatically be set to *True*
- **directory**:
Optional.
Where to save model's hdf5 file. If directory is not *None* and
save is not *False*, then save will automatically be set to *True*

##### Raises:

- *RuntimeWarning*:
If converted and original model aren't identical, and ignore_tests
is False

##### Returns:

If model has been exported to a file, it will return the name of the file.
Else, it returns the converted model.

#### `pytorch_keras_converter.API.convert_and_save`

```
def convert_and_save(model,
                     input_shape,
                     weights=True,
                     quiet=True,
                     ignore_tests=False,
                     input_range=None,
                     filename=None,
                     directory=None):
```

##### Arguments:

- **model**:
A Keras or PyTorch model or layer to convert
- **input_shape** (list, tuple or int):
Input shape, without batchsize.
- **weights** (bool):
Also convert weights. If set to *False*, only convert model
architecture
- **quiet** (bool):
If *False*, display a progress bar and some messages
- **ignore_tests** (bool):
If tests should be ignored. 
  - If set to *False*, converted model will
still be tested by security. If models are not identical, it will
only print a warning.
  - If set to *True*, and models are not identical, *RuntimeWarning* will
be raised
- **input_range**:
Optional.
A list of 2 elements containing max and min values to give as
input to the model when performing the tests. If *None,* models will
be tested on samples from the "standard normal" distribution.
- **filename**:
Optional.
Filename to give to model's hdf5 file. If filename is not *None* and
save is not *False*, then save will automatically be set to *True*
- **directory**:
Optional.
Where to save model's hdf5 file. If directory is not *None* and
save is not *False*, then save will automatically be set to *True*

##### Returns:

Name of created hdf5 file

#### `pytorch_keras_converter.cadene_to_tf`

```
def cadene_to_tf(modelList=None, 
                 outputDirectory=None, 
                 quiet=True):
```

##### Arguments:

- **modelList**:
A *tuple* or *list* of names of the models to convert
OR
A *str* telling the emplacement of a file containing names
of models to convert (one model per line)
If you want a particular config for each model, put it between
parenthesis after model's name, for example:
'se_resnet50(pretrained=None)'
- **outputDirectory** (str):
Optionnal. Where hdf5 files should be saved
- **quiet** (bool):
If *False*, display a progress bar

##### Raises:

- *TypeError*:
If modelList wasn't a *str*, *list*, or *tuple*

##### Returns:

A list of created files

## Supported models and layers

### Supported [Cadene's models](https://github.com/cadene/pretrained-models.pytorch)

- [SE-ResNet50](https://github.com/Cadene/pretrained-models.pytorch#senet)
- [SE-ResNet101](https://github.com/Cadene/pretrained-models.pytorch#senet)
- [SE-ResNet152](https://github.com/Cadene/pretrained-models.pytorch#senet)
- [CaffeResNet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet)
- [BNInception](https://github.com/Cadene/pretrained-models.pytorch#bninception)
- [FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet)
- [ResNet18](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
- [ResNet34](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
- [ResNet50](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
- [ResNet101](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
- [ResNet152](https://github.com/Cadene/pretrained-models.pytorch#torchvision)

### Other models

Some simple PyTorch models are supported. Supported layers are:

- torch.nn.AvgPool2d(count_include_pad=True)
- torch.nn.MaxPool2d(dilation=1)
- torch.nn.Batchnorm2d
- torch.nn.Conv2d(groups=1, padding_mode='zeros')
- torch.nn.Linear
- torch.nn.ZeroPad2d
- torch.nn.Dropout
- torch.nn.Sequential
- torch.nn.ReLU
- torch.nn.Sigmoid
- torch.nn.AdaptiveAvgPool2d(output_size=1)

To convert a custom PyTorch model, use `pytorch_keras_converter.API`.

## Known issues

### Fails to import `pretrainedmodels`

If `pretrainedmodels` module isn't available, you can still convert models manually.

 1. Download `pretrainedmodels` from [GitHub](https://github.com/cadene/pretrained-models.pytorch)
```
git clone https://github.com/Cadene/pretrained-models.pytorch.git
```

 2. Open a Python shell 
```
cd pretrained-models.pytorch
python
```

3. Create a model
```
>>> model = pretrainedmodels.se_resnet50()
>>> input_shape = pretrainedmodels.pretrained_settings['se_resnet50']['input_size']
```

4. Convert your model
```
>>> import pytorch_keras_converter as pkc
>>> pkc.API.convert_and_save(model, input_shape)
```

### Can't run the tests

If you use `pytorch_keras_converter.API` or `pytorch_keras_converter.cadene_to_tf` it will only show a warning : `Warinig: tests unavailable!`.

If you manually test models using `pytorch_keras_converter.tests` it will raise the exception `InvalidArgumentError`.

This is because some TensorFlow layers only support NHWC (i.e channels last) on CPU.

Using Anaconda solves this issue.

## Authors

* [**Alban Benmouffek**](https://github.com/sonibla)

## Code of conduct

This repository is fully [PEP8](https://www.python.org/dev/peps/pep-0008/) compliant.

When reporting issues, please specify your OS, your version of Python, versions of every dependency and if you're using a particular environment (Anaconda for example).

You can use `pip freeze` to see versions of your Python modules.

## License

This project is licensed under the [MIT License](https://tldrlegal.com/license/mit-license) - see the [LICENSE](https://github.com/sonibla/pytorch_keras_converter/blob/master/LICENSE) file for details

## Acknowledgments

* [**Rémi Cadene**](https://github.com/Cadene)
* [**David Picard**](https://github.com/davidpicard)
* [**Pierre Jacob**](https://github.com/pierre-jacob)
