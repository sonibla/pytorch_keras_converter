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

from ... import torch2keras as t2k


def spreadSignal(model):
    if model.type == 'BasicConv2d':
        model.ConnectModelInputToChildren('conv')
        if model.getChild(name='relu') is None:
            model.ConnectLayers('conv', 'bn')
            model.ConnectChildrenOutputToModel('bn')
        else:
            model.ConnectLayers('conv', 'bn', 'relu')
            model.ConnectChildrenOutputToModel('relu')

    elif model.type == 'Mixed_3a':
        model.ConnectModelInputToChildren('conv')
        model.ConnectModelInputToChildren('maxpool')
        model.ConnectChildrenOutputToModel('maxpool',
                                           'conv')

    elif model.type == 'Mixed_4a':
        model.ConnectModelInputToChildren('branch0')
        model.ConnectModelInputToChildren('branch1')
        model.ConnectChildrenOutputToModel('branch0',
                                           'branch1')

    elif model.type == 'Mixed_5a':
        model.ConnectModelInputToChildren('conv')
        model.ConnectModelInputToChildren('maxpool')
        model.ConnectChildrenOutputToModel('conv',
                                           'maxpool')

    elif model.type == 'Inception_A':
        model.ConnectModelInputToChildren('branch0')
        model.ConnectModelInputToChildren('branch1')
        model.ConnectModelInputToChildren('branch2')
        model.ConnectModelInputToChildren('branch3')
        model.ConnectChildrenOutputToModel('branch0',
                                           'branch1',
                                           'branch2',
                                           'branch3')

    elif model.type == 'Reduction_A':
        model.ConnectModelInputToChildren('branch0')
        model.ConnectModelInputToChildren('branch1')
        model.ConnectModelInputToChildren('branch2')
        model.ConnectChildrenOutputToModel('branch0',
                                           'branch1',
                                           'branch2')

    elif model.type == 'Inception_B':
        model.ConnectModelInputToChildren('branch0')
        model.ConnectModelInputToChildren('branch1')
        model.ConnectModelInputToChildren('branch2')
        model.ConnectModelInputToChildren('branch3')
        model.ConnectChildrenOutputToModel('branch0',
                                           'branch1',
                                           'branch2',
                                           'branch3')

    elif model.type == 'Reduction_B':
        model.ConnectModelInputToChildren('branch0')
        model.ConnectModelInputToChildren('branch1')
        model.ConnectModelInputToChildren('branch2')
        model.ConnectChildrenOutputToModel('branch0',
                                           'branch1',
                                           'branch2')

    elif model.type == 'Inception_C':
        model.ConnectModelInputToChildren('branch0')
        model.ConnectModelInputToChildren('branch1_0')
        model.ConnectModelInputToChildren('branch2_0')
        model.ConnectModelInputToChildren('branch3')
        model.ConnectLayers('branch1_0', 'branch1_1a')
        model.ConnectLayers('branch1_0', 'branch1_1b')
        model.ConnectLayers('branch2_0', 'branch2_1', 'branch2_2')
        model.ConnectLayers('branch2_2', 'branch2_3a')
        model.ConnectLayers('branch2_2', 'branch2_3b')
        model.ConnectChildrenOutputToModel('branch0',
                                           'branch1_1a',
                                           'branch1_1b',
                                           'branch2_3a',
                                           'branch2_3b',
                                           'branch3')

    elif model.type == 'InceptionV4':
        model.ConnectModelInputToChildren('features')

        featuresOut = model.getChild(name='features').kerasOutput

        if featuresOut is not None:
            adaptiveAvgPoolWidth = model['features'].output_shape[2]

            KerasAvgPool = keras.layers.AveragePooling2D

            avgPool = KerasAvgPool(pool_size=adaptiveAvgPoolWidth,
                                   padding='valid',
                                   data_format='channels_first',
                                   input_shape=model['features'].output_shape)

            avgPoolOut = avgPool(featuresOut)

            shapeOUT = t2k.kerasShape(avgPoolOut)

            flatten = keras.layers.Flatten(data_format='channels_first',
                                           input_shape=shapeOUT)
            flattenOut = flatten(avgPoolOut)

            model.getChild(name='last_linear').kerasInput = flattenOut

        model.Connect2Layers('features', 'last_linear', connectKeras=False)

        model.ConnectChildrenOutputToModel('last_linear')

    elif model.type == 'Sequential':
        model.ConnectModelInputToChildren('0')
        for i in range(len(model.children)-1):
            model.Connect2Layers(str(i), str(i+1))
        model.ConnectChildrenOutputToModel(str(len(model.children)-1))

    else:
        err = "Warning: layer or model '{}' not recognized!".format(model.type)
        raise NotImplementedError(err)
