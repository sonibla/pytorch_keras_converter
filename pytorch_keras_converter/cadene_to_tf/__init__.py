"""
API to convert Cadene's models

Supported models:
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

Convert and save models with cadene_to_tf

Examples:
    cadene_to_tf(modelList=['se_resnet50'], quiet=False)

    cadene_to_tf(modelList=['se_resnet50',
                            'bninception(pretrained=None)])

    cadene_to_tf(modelList=models.txt)

"""
from .cadene_to_tf import cadene_to_tf
