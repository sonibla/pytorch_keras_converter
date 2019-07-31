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


def spreadSignal(model):

    def Output(child):
        if isinstance(child, str) and model.getChild(name=child) is not None:
            return model.getChild(name=child).kerasOutput
        elif child in model.children:
            return child.kerasOutput
        return None

    if model.type == 'SEModule':
        model.ConnectModelInputToChildren('avg_pool')
        model.ConnectLayers('avg_pool',
                            'fc1',
                            'relu',
                            'fc2',
                            'sigmoid')

        times = keras.layers.Multiply()

        model.kerasOutput = times([model.kerasInput, Output('sigmoid')])

    elif model.type in ['Bottleneck',
                        'SEBottleneck',
                        'SEResNetBottleneck',
                        'SEResNeXtBottleneck']:

        model.ConnectModelInputToChildren('conv1')
        model.ConnectLayers('conv1',
                            'bn1')

        if Output('bn1') is not None:
            relu1 = keras.layers.ReLU(input_shape=model['bn1'].output_shape)
            outRelu1 = relu1(Output('bn1'))
            model.getChild(name='conv2').kerasInput = outRelu1

        model.ConnectLayers('conv2',
                            'bn2')

        if Output('bn2') is not None:
            relu2 = keras.layers.ReLU(input_shape=model['bn2'].output_shape)
            outRelu2 = relu2(Output('bn2'))
            model.getChild(name='conv3').kerasInput = outRelu2

        model.ConnectLayers('conv3',
                            'bn3',
                            'se_module')

        if model.getChild(name='downsample') is not None:
            model.ConnectModelInputToChildren('downsample')
            if Output('downsample') is not None and \
                    Output('se_module') is not None:

                add = keras.layers.Add()
                out = add([Output('se_module'), Output('downsample')])
            else:
                out = None
        else:
            if Output('se_module') is not None:
                add = keras.layers.Add()
                out = add([Output('se_module'), model.kerasInput])
            else:
                out = None

        if out is not None:
            relu3 = keras.layers.ReLU()
            outRelu3 = relu3(out)
            model.kerasOutput = outRelu3

    elif model.type == 'SENet':
        model.ConnectModelInputToChildren('layer0')
        model.ConnectLayers('layer0',
                            'layer1',
                            'layer2',
                            'layer3',
                            'layer4',
                            'avg_pool')

        if model.getChild(name='dropout') is not None:
            model.ConnectLayers('avg_pool', 'dropout')
            featuresOut = Output('dropout')
            model.Connect2Layers('dropout', 'last_linear', connectKeras=False)
        else:
            featuresOut = Output('avg_pool')
            model.Connect2Layers('avg_pool', 'last_linear', connectKeras=False)

        if featuresOut is not None:
            flatten = keras.layers.Flatten(data_format='channels_first')

            flattenOut = flatten(featuresOut)

            model.getChild(name='last_linear').kerasInput = flattenOut

        model.ConnectChildrenOutputToModel('last_linear')

    elif model.type == 'Sequential':
        model.ConnectModelInputToChildren('0')
        for i in range(len(model.children)-1):
            model.Connect2Layers(str(i), str(i+1))
        model.ConnectChildrenOutputToModel(str(len(model.children)-1))

    else:
        err = "Warning: layer or model '{}' not recognized!".format(model.type)
        raise NotImplementedError(err)
