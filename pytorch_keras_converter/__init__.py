"""
Conversion Utility
Convert and analyse Keras and pyTorch models

Dependencies:
dependencies = ['torch',
                'pretrainedmodels',
                'tensorflow',
                'graphviz',
                'numpy',
                'h5py',
                'tqdm']


Features:

#########################
####- CONVERSIONS:  -####
#########################
    -pyTorch -> Keras:
        simple models and Cadene's models only. You can use
        original weights or random ones. Converting weights takes a lot of time
        because of Keras' set_weights function

        To easily convert Cadene models see cadene_to_tf.py

        Supported Cadene's models:
            -se_resnet50
            -se_resnet101
            -se_resnet152
            -cafferesnet101
            -bninception
            -fbresnet152
            -resnet18
            -resnet34
            -resnet50
            -resnet101
            -resnet152

        Supported layers:
            -torch.nn.AvgPool2d(count_include_pad = True)
            -torch.nn.MaxPool2d(dilation = 1)
            -torch.nn.Batchnorm2d
            -torch.nn.Conv2d(groups = 1, padding_mode = 'zeros')
            -torch.nn.Linear
            -torch.nn.ZeroPad2d
            -torch.nn.Dropout
            -torch.nn.Sequential
            -torch.nn.ReLU
            -torch.nn.Sigmoid
            -torch.nn.AdaptiveAvgPool2d(output_size = 1)

    -Keras -> pyTorch : NOT IMPLEMENTED

#################################
####- DEV AND DEBUG TOOLS:  -####
#################################

    -pyTorch layers listing : OK. See utility/LayerRepresentation.py for
        details (LayerRepresentation.summary)

    -Keras layers listing : OK. See utility/LayerRepresentation.py for details
    (LayerRepresentation.summary).

    -pyTorch DOT graph rendering/export : OK, but edges aren't reliable. See
        utility/LayerRepresentation.py for details (LayerRepresentation.DOT).

    -Keras DOT graph rendering/export : OK, but edges aren't reliable. See
        utility/LayerRepresentation.py for details (LayerRepresentation.DOT).

    -Saving a Keras model to a .py file (architecture only):
        OK, only if original model was a simple pyTorch model. See
        utility/core.py for details.

    -Saing an entire model after pyTorch to Keras conversion :
        OK. See API.py for details.

    -Comparing 2 models (Keras, pyTorch) on random data :
        OK (mean squared error, progress bar available, severak tests
        available). See tests.py for details.

"""

name = "pytorch_keras_converter"
supported_cadene_models = ['se_resnet50',
                           'se_resnet101',
                           'se_resnet152',
                           'cafferesnet101',
                           'bninception',
                           'fbresnet152',
                           'resnet18',
                           'resnet34',
                           'resnet50',
                           'resnet101',
                           'resnet152']

from . import API
from .cadene_to_tf import cadene_to_tf
from . import utility
