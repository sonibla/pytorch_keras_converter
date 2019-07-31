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

    if model.type == 'BasicBlock':
        model.ConnectModelInputToChildren('conv1')
        model.ConnectLayers('conv1',
                            'bn1')

        if Output('bn1') is not None:
            relu1 = keras.layers.ReLU(input_shape=model['bn1'].output_shape)
            outRelu1 = relu1(Output('bn1'))
            model.getChild(name='conv2').kerasInput = outRelu1

        model.ConnectLayers('conv2',
                            'bn2')

        if model.getChild(name='downsample') is not None:
            model.ConnectModelInputToChildren('downsample')
            if Output('downsample') is not None and Output('bn2') is not None:
                add = keras.layers.Add()
                out = add([Output('bn2'), Output('downsample')])
            else:
                out = None
        else:
            if Output('bn2') is not None:
                add = keras.layers.Add()
                out = add([Output('bn2'), model.kerasInput])
            else:
                out = None

        if out is not None:
            relu2 = keras.layers.ReLU()
            outRelu2 = relu2(out)
            model.kerasOutput = outRelu2

    elif model.type == 'Bottleneck':
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
                            'bn3')

        if model.getChild(name='downsample') is not None:
            model.ConnectModelInputToChildren('downsample')
            if Output('downsample') is not None and Output('bn3') is not None:
                add = keras.layers.Add()
                out = add([Output('bn3'), Output('downsample')])
            else:
                out = None
        else:
            if Output('bn3') is not None:
                add = keras.layers.Add()
                out = add([Output('bn3'), model.kerasInput])
            else:
                out = None

        if out is not None:
            relu3 = keras.layers.ReLU()
            outRelu3 = relu3(out)
            model.kerasOutput = outRelu3

    elif model.type == 'ResNet':
        model.ConnectModelInputToChildren('conv1')
        model.ConnectLayers('conv1',
                            'bn1',
                            'relu',
                            'maxpool',
                            'layer1',
                            'layer2',
                            'layer3',
                            'layer4',
                            'avgpool')

        featuresOut = Output('avgpool')
        model.Connect2Layers('avgpool', 'last_linear', connectKeras=False)

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
