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

    def ConcatenateOutputs(*names):
        model.ConnectChildrenOutputToModel(*names, connectKeras=False)
        outputs = list()
        for i in range(len(names)):
            if not isinstance(names[i], str):
                outputs.append(names[i])
            elif Output(names[i]) is None:
                return None
            else:
                outputs.append(Output(names[i]))
        return keras.layers.concatenate(outputs, axis=1)

    def Output(child):
        if isinstance(child, str) and model.getChild(name=child) is not None:
            return model.getChild(name=child).kerasOutput
        elif child in model.children:
            return child.kerasOutput
        return None

    if model.type == 'BNInception':
        model.ConnectModelInputToChildren('conv1_7x7_s2')
        model.ConnectLayers('conv1_7x7_s2',
                            'conv1_7x7_s2_bn',
                            'conv1_relu_7x7',
                            'pool1_3x3_s2',
                            'conv2_3x3_reduce',
                            'conv2_3x3_reduce_bn',
                            'conv2_relu_3x3_reduce',
                            'conv2_3x3',
                            'conv2_3x3_bn',
                            'conv2_relu_3x3',
                            'pool2_3x3_s2')

        model.ConnectLayers('pool2_3x3_s2', 'inception_3a_1x1')
        model.ConnectLayers('pool2_3x3_s2', 'inception_3a_3x3_reduce')
        model.ConnectLayers('pool2_3x3_s2', 'inception_3a_double_3x3_reduce')
        model.ConnectLayers('pool2_3x3_s2', 'inception_3a_pool')

        model.ConnectLayers('inception_3a_1x1',
                            'inception_3a_1x1_bn',
                            'inception_3a_relu_1x1')

        model.ConnectLayers('inception_3a_3x3_reduce',
                            'inception_3a_3x3_reduce_bn',
                            'inception_3a_relu_3x3_reduce',
                            'inception_3a_3x3',
                            'inception_3a_3x3_bn',
                            'inception_3a_relu_3x3')

        model.ConnectLayers('inception_3a_double_3x3_reduce',
                            'inception_3a_double_3x3_reduce_bn',
                            'inception_3a_relu_double_3x3_reduce',
                            'inception_3a_double_3x3_1',
                            'inception_3a_double_3x3_1_bn',
                            'inception_3a_relu_double_3x3_1',
                            'inception_3a_double_3x3_2',
                            'inception_3a_double_3x3_2_bn',
                            'inception_3a_relu_double_3x3_2')

        model.ConnectLayers('inception_3a_pool',
                            'inception_3a_pool_proj',
                            'inception_3a_pool_proj_bn',
                            'inception_3a_relu_pool_proj')

        childrenNames = ['inception_3b_1x1',
                         'inception_3b_3x3_reduce',
                         'inception_3b_double_3x3_reduce',
                         'inception_3b_pool']

        model.ConnectChildrenOutputToChildren('inception_3a_relu_1x1',
                                              'inception_3a_relu_3x3',
                                              'inception_3a_relu_double_3x3_2',
                                              'inception_3a_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_3b_1x1',
                            'inception_3b_1x1_bn',
                            'inception_3b_relu_1x1')

        model.ConnectLayers('inception_3b_3x3_reduce',
                            'inception_3b_3x3_reduce_bn',
                            'inception_3b_relu_3x3_reduce',
                            'inception_3b_3x3',
                            'inception_3b_3x3_bn',
                            'inception_3b_relu_3x3')

        model.ConnectLayers('inception_3b_double_3x3_reduce',
                            'inception_3b_double_3x3_reduce_bn',
                            'inception_3b_relu_double_3x3_reduce',
                            'inception_3b_double_3x3_1',
                            'inception_3b_double_3x3_1_bn',
                            'inception_3b_relu_double_3x3_1',
                            'inception_3b_double_3x3_2',
                            'inception_3b_double_3x3_2_bn',
                            'inception_3b_relu_double_3x3_2')

        model.ConnectLayers('inception_3b_pool',
                            'inception_3b_pool_proj',
                            'inception_3b_pool_proj_bn',
                            'inception_3b_relu_pool_proj')

        childrenNames = ['inception_3c_3x3_reduce',
                         'inception_3c_double_3x3_reduce',
                         'inception_3c_pool']

        model.ConnectChildrenOutputToChildren('inception_3b_relu_1x1',
                                              'inception_3b_relu_3x3',
                                              'inception_3b_relu_double_3x3_2',
                                              'inception_3b_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_3c_3x3_reduce',
                            'inception_3c_3x3_reduce_bn',
                            'inception_3c_relu_3x3_reduce',
                            'inception_3c_3x3',
                            'inception_3c_3x3_bn',
                            'inception_3c_relu_3x3')

        model.ConnectLayers('inception_3c_double_3x3_reduce',
                            'inception_3c_double_3x3_reduce_bn',
                            'inception_3c_relu_double_3x3_reduce',
                            'inception_3c_double_3x3_1',
                            'inception_3c_double_3x3_1_bn',
                            'inception_3c_relu_double_3x3_1',
                            'inception_3c_double_3x3_2',
                            'inception_3c_double_3x3_2_bn',
                            'inception_3c_relu_double_3x3_2')

        childrenNames = ['inception_4a_1x1',
                         'inception_4a_3x3_reduce',
                         'inception_4a_double_3x3_reduce',
                         'inception_4a_pool']

        model.ConnectChildrenOutputToChildren('inception_3c_relu_3x3',
                                              'inception_3c_relu_double_3x3_2',
                                              'inception_3c_pool',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_4a_1x1',
                            'inception_4a_1x1_bn',
                            'inception_4a_relu_1x1')

        model.ConnectLayers('inception_4a_3x3_reduce',
                            'inception_4a_3x3_reduce_bn',
                            'inception_4a_relu_3x3_reduce',
                            'inception_4a_3x3',
                            'inception_4a_3x3_bn',
                            'inception_4a_relu_3x3')

        model.ConnectLayers('inception_4a_double_3x3_reduce',
                            'inception_4a_double_3x3_reduce_bn',
                            'inception_4a_relu_double_3x3_reduce',
                            'inception_4a_double_3x3_1',
                            'inception_4a_double_3x3_1_bn',
                            'inception_4a_relu_double_3x3_1',
                            'inception_4a_double_3x3_2',
                            'inception_4a_double_3x3_2_bn',
                            'inception_4a_relu_double_3x3_2')

        model.ConnectLayers('inception_4a_pool',
                            'inception_4a_pool_proj',
                            'inception_4a_pool_proj_bn',
                            'inception_4a_relu_pool_proj')

        childrenNames = ['inception_4b_1x1',
                         'inception_4b_3x3_reduce',
                         'inception_4b_double_3x3_reduce',
                         'inception_4b_pool']

        model.ConnectChildrenOutputToChildren('inception_4a_relu_1x1',
                                              'inception_4a_relu_3x3',
                                              'inception_4a_relu_double_3x3_2',
                                              'inception_4a_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_4b_1x1',
                            'inception_4b_1x1_bn',
                            'inception_4b_relu_1x1')

        model.ConnectLayers('inception_4b_3x3_reduce',
                            'inception_4b_3x3_reduce_bn',
                            'inception_4b_relu_3x3_reduce',
                            'inception_4b_3x3',
                            'inception_4b_3x3_bn',
                            'inception_4b_relu_3x3')

        model.ConnectLayers('inception_4b_double_3x3_reduce',
                            'inception_4b_double_3x3_reduce_bn',
                            'inception_4b_relu_double_3x3_reduce',
                            'inception_4b_double_3x3_1',
                            'inception_4b_double_3x3_1_bn',
                            'inception_4b_relu_double_3x3_1',
                            'inception_4b_double_3x3_2',
                            'inception_4b_double_3x3_2_bn',
                            'inception_4b_relu_double_3x3_2')

        model.ConnectLayers('inception_4b_pool',
                            'inception_4b_pool_proj',
                            'inception_4b_pool_proj_bn',
                            'inception_4b_relu_pool_proj')

        childrenNames = ['inception_4c_1x1',
                         'inception_4c_3x3_reduce',
                         'inception_4c_double_3x3_reduce',
                         'inception_4c_pool']

        model.ConnectChildrenOutputToChildren('inception_4b_relu_1x1',
                                              'inception_4b_relu_3x3',
                                              'inception_4b_relu_double_3x3_2',
                                              'inception_4b_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_4c_1x1',
                            'inception_4c_1x1_bn',
                            'inception_4c_relu_1x1')

        model.ConnectLayers('inception_4c_3x3_reduce',
                            'inception_4c_3x3_reduce_bn',
                            'inception_4c_relu_3x3_reduce',
                            'inception_4c_3x3',
                            'inception_4c_3x3_bn',
                            'inception_4c_relu_3x3')

        model.ConnectLayers('inception_4c_double_3x3_reduce',
                            'inception_4c_double_3x3_reduce_bn',
                            'inception_4c_relu_double_3x3_reduce',
                            'inception_4c_double_3x3_1',
                            'inception_4c_double_3x3_1_bn',
                            'inception_4c_relu_double_3x3_1',
                            'inception_4c_double_3x3_2',
                            'inception_4c_double_3x3_2_bn',
                            'inception_4c_relu_double_3x3_2')

        model.ConnectLayers('inception_4c_pool',
                            'inception_4c_pool_proj',
                            'inception_4c_pool_proj_bn',
                            'inception_4c_relu_pool_proj')

        childrenNames = ['inception_4d_1x1',
                         'inception_4d_3x3_reduce',
                         'inception_4d_double_3x3_reduce',
                         'inception_4d_pool']

        model.ConnectChildrenOutputToChildren('inception_4c_relu_1x1',
                                              'inception_4c_relu_3x3',
                                              'inception_4c_relu_double_3x3_2',
                                              'inception_4c_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_4d_1x1',
                            'inception_4d_1x1_bn',
                            'inception_4d_relu_1x1')

        model.ConnectLayers('inception_4d_3x3_reduce',
                            'inception_4d_3x3_reduce_bn',
                            'inception_4d_relu_3x3_reduce',
                            'inception_4d_3x3',
                            'inception_4d_3x3_bn',
                            'inception_4d_relu_3x3')

        model.ConnectLayers('inception_4d_double_3x3_reduce',
                            'inception_4d_double_3x3_reduce_bn',
                            'inception_4d_relu_double_3x3_reduce',
                            'inception_4d_double_3x3_1',
                            'inception_4d_double_3x3_1_bn',
                            'inception_4d_relu_double_3x3_1',
                            'inception_4d_double_3x3_2',
                            'inception_4d_double_3x3_2_bn',
                            'inception_4d_relu_double_3x3_2')

        model.ConnectLayers('inception_4d_pool',
                            'inception_4d_pool_proj',
                            'inception_4d_pool_proj_bn',
                            'inception_4d_relu_pool_proj')

        childrenNames = ['inception_4e_3x3_reduce',
                         'inception_4e_double_3x3_reduce',
                         'inception_4e_pool']

        model.ConnectChildrenOutputToChildren('inception_4d_relu_1x1',
                                              'inception_4d_relu_3x3',
                                              'inception_4d_relu_double_3x3_2',
                                              'inception_4d_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_4e_3x3_reduce',
                            'inception_4e_3x3_reduce_bn',
                            'inception_4e_relu_3x3_reduce',
                            'inception_4e_3x3',
                            'inception_4e_3x3_bn',
                            'inception_4e_relu_3x3')

        model.ConnectLayers('inception_4e_double_3x3_reduce',
                            'inception_4e_double_3x3_reduce_bn',
                            'inception_4e_relu_double_3x3_reduce',
                            'inception_4e_double_3x3_1',
                            'inception_4e_double_3x3_1_bn',
                            'inception_4e_relu_double_3x3_1',
                            'inception_4e_double_3x3_2',
                            'inception_4e_double_3x3_2_bn',
                            'inception_4e_relu_double_3x3_2')

        childrenNames = ['inception_5a_1x1',
                         'inception_5a_3x3_reduce',
                         'inception_5a_double_3x3_reduce',
                         'inception_5a_pool']

        model.ConnectChildrenOutputToChildren('inception_4e_relu_3x3',
                                              'inception_4e_relu_double_3x3_2',
                                              'inception_4e_pool',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_5a_1x1',
                            'inception_5a_1x1_bn',
                            'inception_5a_relu_1x1')

        model.ConnectLayers('inception_5a_3x3_reduce',
                            'inception_5a_3x3_reduce_bn',
                            'inception_5a_relu_3x3_reduce',
                            'inception_5a_3x3',
                            'inception_5a_3x3_bn',
                            'inception_5a_relu_3x3')

        model.ConnectLayers('inception_5a_double_3x3_reduce',
                            'inception_5a_double_3x3_reduce_bn',
                            'inception_5a_relu_double_3x3_reduce',
                            'inception_5a_double_3x3_1',
                            'inception_5a_double_3x3_1_bn',
                            'inception_5a_relu_double_3x3_1',
                            'inception_5a_double_3x3_2',
                            'inception_5a_double_3x3_2_bn',
                            'inception_5a_relu_double_3x3_2')

        model.ConnectLayers('inception_5a_pool',
                            'inception_5a_pool_proj',
                            'inception_5a_pool_proj_bn',
                            'inception_5a_relu_pool_proj')

        childrenNames = ['inception_5b_1x1',
                         'inception_5b_3x3_reduce',
                         'inception_5b_double_3x3_reduce',
                         'inception_5b_pool']

        model.ConnectChildrenOutputToChildren('inception_5a_relu_1x1',
                                              'inception_5a_relu_3x3',
                                              'inception_5a_relu_double_3x3_2',
                                              'inception_5a_relu_pool_proj',
                                              childrenNames=childrenNames)

        model.ConnectLayers('inception_5b_1x1',
                            'inception_5b_1x1_bn',
                            'inception_5b_relu_1x1')

        model.ConnectLayers('inception_5b_3x3_reduce',
                            'inception_5b_3x3_reduce_bn',
                            'inception_5b_relu_3x3_reduce',
                            'inception_5b_3x3',
                            'inception_5b_3x3_bn',
                            'inception_5b_relu_3x3')

        model.ConnectLayers('inception_5b_double_3x3_reduce',
                            'inception_5b_double_3x3_reduce_bn',
                            'inception_5b_relu_double_3x3_reduce',
                            'inception_5b_double_3x3_1',
                            'inception_5b_double_3x3_1_bn',
                            'inception_5b_relu_double_3x3_1',
                            'inception_5b_double_3x3_2',
                            'inception_5b_double_3x3_2_bn',
                            'inception_5b_relu_double_3x3_2')

        model.ConnectLayers('inception_5b_pool',
                            'inception_5b_pool_proj',
                            'inception_5b_pool_proj_bn',
                            'inception_5b_relu_pool_proj')

        featuresOut = ConcatenateOutputs('inception_5b_relu_1x1',
                                         'inception_5b_relu_3x3',
                                         'inception_5b_relu_double_3x3_2',
                                         'inception_5b_relu_pool_proj')

        if featuresOut is not None:
            output_shape = t2k.kerasShape(featuresOut)

            adaptAvgPoolWidth = output_shape[2]
            avgPl = keras.layers.AveragePooling2D(pool_size=adaptAvgPoolWidth,
                                                  padding='valid',
                                                  data_format='channels_first')

            avgPoolOut = avgPl(featuresOut)

            flatten = keras.layers.Flatten(data_format='channels_first')

            flattenOut = flatten(avgPoolOut)

            model.getChild(name='last_linear').kerasInput = flattenOut

        model.ConnectChildrenOutputToChild('inception_5b_relu_1x1',
                                           'inception_5b_relu_3x3',
                                           'inception_5b_relu_double_3x3_2',
                                           'inception_5b_relu_pool_proj',
                                           childName='last_linear',
                                           connectKeras=False)

        model.ConnectChildrenOutputToModel('last_linear')

    elif model.type == 'Sequential':
        model.ConnectModelInputToChildren('0')
        for i in range(len(model.children)-1):
            model.Connect2Layers(str(i), str(i+1))
        model.ConnectChildrenOutputToModel(str(len(model.children)-1))

    else:
        err = "Warning: layer or model '{}' not recognized!".format(model.type)
        raise NotImplementedError(err)
