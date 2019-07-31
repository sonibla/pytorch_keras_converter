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
try:
    import graphviz  # Useful in DOT graphs creation and export
except ImportError:
    graphviz = None

from . import torch2keras as t2k


def UsedFramework(module):
    """
    Tells which framework module is based on.

    Arguments:
        -module:
            The object to test

    Returns:
        -'keras' if module is a keras object
        -'torch' if module is a pyTorch object
        -None if none of them
    """

    moduleClass = str(module.__class__)[8:-2]
    IsKeras = 'keras' in moduleClass
    IsTorch = 'torch' in moduleClass
    if '_backend' in dir(module):
        IsTorch = IsTorch or 'torch' in str(module._backend)

    if IsKeras:
        return 'keras'
    if IsTorch:
        return 'torch'
    return None


def normalizeShape(shape):
    """
    Method used to convert shape to tuple

    Arguments:
        -shape:
            int, tuple, list, or anything convertible to tuple

    Raises:
        -TypeError:
            If conversion to tuple failed

    Returns:
        A tuple representing the shape
    """
    if isinstance(shape, tuple):
        normalizedShape = shape
    elif isinstance(shape, int):
        normalizedShape = (shape,)
    else:
        try:
            normalizedShape = tuple(shape)
        except TypeError:
            # It was too difficult.
            # This case will never happen if the API is used correctly
            raise TypeError("Could not convert provided shape to tulpe")
    return normalizedShape


class LayerRepresentation:
    """
    Easy-to-use representation of a layer.

    When you create an object of this class, all sub layers (also called
    children) are automatically decomposed and referenced in self.children

    This is useful to go through a model considering only simple layers one
    by one

    Argument:
        -module:
            A PyTorch or Keras model or layer to import
    """

    # --- CORE METHODS ---

    def __init__(self, module=None):

        # --- INITIALIZATIONS ---
        self.parent = None
        # parent represents the block in which the layer is (for example a
        # Sequential)

        self.children = list()
        # children is a list of direct sub layers (for example if module is a
        # Sequential block)

        self.originalFramework = None
        # Framework of module ('keras' or 'torch'). Stays None if provided
        # model isn't compatible

        self.name = str(module.__class__.__name__)
        # name will change if module has been created by recursivity (with
        # addChild method, see below for details)
        self.type = str(module.__class__.__name__)
        self.detailedType = str(module.__class__)[8:-2]
        self.detailedTypeList = self.detailedType.split('.')

        self.equivalent = dict()
        self.equivalentTxt = dict()
        # These are dictionnaries containing equivalents of the layer (or
        # model) in PyTorch or Keras.
        # self.equivalent['torch'] is a PyTorch object (if exists)
        # self.equivalent['keras'] is a Keras object (if exists)
        # self.equivalentTxt contains equivalents that can be put directly in a
        # python file

        self.input_shape = None
        self.output_shape = None
        # In/Output shapes, without batchsize, channels first (tuple)
        # For example : (3, 299, 299)

        self.InputConnectedTo = set()
        # set telling where the input comes from.
        # If the input comes from the input of self.parent (for example if the
        # layer is the first one in a Sequential block), then 0 will be in this
        # set
        # If the input comes from the output of a brother (for example if the
        # layer is in a Sequential block but not in first place), then
        # LayerRepresentation of this brother will be in this set

        self.OutputConnectedTo = set()
        # Same, for output.
        # set telling where the output goes to.
        # If the output goes to the output of self.parent (for example if the
        # layer is the last one in a Sequential block), then 0 will be in this
        # set
        # If the output goes to the input of a brother (for example if the
        # layer is in a Sequential block but not at last place), then
        # LayerRepresentation of this brother will be in this set

        # Initializing self.framework, self.children and self.equivalent :

        framework = UsedFramework(module)

        if framework == 'torch':
            # PyTorch !

            self.originalFramework = 'torch'
            # We don't want to modify the model when converting it
            self.equivalent['torch'] = module.eval()

            # listing each sub layers (children)
            children = dict(module.named_children()).items()
            for name, child in children:
                # Hidden recursivity (see addChild method for details)
                self.addChild(child, name=name)

        elif framework == 'keras':
            # Keras !

            self.originalFramework = 'keras'
            # We don't want to modify the model when converting it
            if keras is None:
                # But Keras isn't available so we can't clone the model...
                self.equivalent['keras'] = module
            else:
                self.equivalent['keras'] = keras.models.clone_model(module)

            # listing each sub layers (children)
            if 'layers' in dir(module):
                for child in module.layers:
                    # Hidden recursivity (see addChild method for details)
                    self.addChild(child, name=child.name)

        self.kerasOutput = None
        self.kerasInput = None
        # During a PyTorch to Keras conversion, if Keras Functionnal API is
        # used, inputs and outputs are stored here.

    def __setattr__(self, attr, val):
        object.__setattr__(self, attr, val)

        if 'kerasInput' in dir(self):
            # We have to check this first (if kerasInput isn't in dir(self), it
            # means that we're still in self.__init__, we should not interfere
            # with it)

            if attr == 'input_shape':
                # If input_shape was modified, we check that it's a tuple
                if not isinstance(self.input_shape, tuple):
                    self.input_shape = normalizeShape(self.input_shape)

            elif attr == 'output_shape':
                # If output_shape was modified, we check that it's a tuple
                if not isinstance(self.output_shape, tuple):
                    self.output_shape = normalizeShape(self.output_shape)

            elif attr == 'kerasInput':
                # If kerasInput was modified, we automatically compute (or
                # update) self.equivalent['keras'] or self.kerasOutput, if
                # possible
                inputExist = self.kerasInput is not None
                outputExist = self.kerasOutput is not None
                equivExist = 'keras' in self.equivalent.keys()

                if inputExist and equivExist:
                    # We can compute (update or create) kerasOutput !
                    output = self.equivalent['keras'](self.kerasInput)
                    self.kerasOutput = output

                if inputExist and outputExist and not(equivExist):
                    # We can compute (create) keras equivalent !
                    if keras is None:
                        err = "Could not import keras. Conversion failed !"
                        raise ImportError(err)
                    kerasEq = keras.models.Model(inputs=self.kerasInput,
                                                 outputs=self.kerasOutput,
                                                 name=self.name)
                    self.equivalent['keras'] = kerasEq

            if self.kerasInput is not None and self.input_shape is not None:
                # Here, we check that input computed with Keras Funcitonnal API
                # have the correct shape (i.e self.input_shape).
                # If shapes are not the same, it means that the conversion
                # failed, and all we have to do is raising a RuntimeError to
                # warn the user about it

                shape = self.kerasInputShape()

                if shape is not None and shape != self.input_shape:
                    err = "Conversion failed! Details: at layer {}, input \
shape should be {}, but is {}\
                    ".format(self.name, self.input_shape, shape)

                    raise RuntimeError(err)

            if self.kerasOutput is not None and self.output_shape is not None:
                # Here, we check that output computed with Keras Funcitonnal
                # API have the correct shape (i.e self.output_shape).
                # If shapes are not the same, it means that the conversion
                # failed, and all we have to do is raising a RuntimeError to
                # warn the user about it

                shape = self.kerasOutputShape()

                if shape is not None and shape != self.output_shape:
                    err = "Conversion failed! Details: at layer {}, output \
shape should be {}, but is {}\
                    ".format(self.name, self.output_shape, shape)

                    raise RuntimeError(err)

    # --- FAMILY METHODS ---

    def __getitem__(self, index):
        """
        Equivalent of self.getChild if index is str

        Equivalent of self.getChildId with framework=None if index is int
        """
        if isinstance(index, str):
            return self.getChild(name=index)
        if isinstance(index, int):
            return self.getChildId(identifier=index)
        return None

    def getChildId(self, identifier=None, framework=None):
        """
        If framework is None: search for a layer which id is identifier
        Else: search for a layer which framework equivalent's id is identifier

        If no child was found, search in the whole model
        Return None if no layer was found at all

        Arguments:
            -identifier
            -framework

        Returns:
            The LayerRepresentation of corresponding layer if this layer exists
        """
        if framework is None:
            for child in self.children:
                if id(child) == identifier:
                    return child
            if id(self) == identifier:
                return self
            mainParent = self.firstParent()
            if id(mainParent) == identifier:
                return mainParent
            for child in mainParent.allChildren():
                if id(child) == identifier:
                    return child
            return None

        else:
            # framework is not None : look for equivalents

            for child in self.children:
                # Look at first in self.children
                if framework in child.equivalent.keys():
                    equiv = child.equivalent[framework]
                    if id(equiv) == identifier:
                        return child

            if framework in self.equivalent.keys():
                if id(self.equivalent[framework]) == identifier:
                    return self
            mainParent = self.firstParent()
            if framework in mainParent.equivalent.keys():
                if id(mainParent.equivalent[framework]) == identifier:
                    return mainParent

            for child in mainParent.allChildren():
                # Look in the entire model
                if framework in child.equivalent.keys():
                    equiv = child.equivalent[framework]
                    if id(equiv) == identifier:
                        return child
            return None

    def getChild(self, name=None):
        """
        Return child which name is name (argument)
        Return None if no child was found
        """
        for child in self.children:
            if child.name == name:
                return child
        return None

    def addChild(self, childEq, name=None):
        """
        Adds a child in self.children

        Arguments:
            -childEq:
                Keras or PyTorch object representign the layer to add
            -name:
                Optional name of the child
        """
        child = LayerRepresentation(childEq)
        child.name = str(name)

        child.parent = self
        self.children.append(child)

        return child

    def delChildren(self):
        """
        Delete every children
        """
        self.children = list()

    def delChild(self, name=None):
        """
        Delete one child, identified by his name
        """
        if self.getChild(name=name) is not None:
            del self.children[self.getChild(name=name)]

    def allChildren(self):
        """
        Returns a list of every child contained in self.children, and their
        sub-layers (using recursivity)
        """
        if not self.children:
            return list()
        else:
            List = self.children
            for child in self.children:
                List = List + child.allChildren()
            return List

    def numberOfChildren(self):
        """
        Uses recursivity to compute total number of sub layers (children)
        contained in the model
        """
        number = len(self.children)
        for child in self.children:
            number += child.numberOfChildren()
        return number

    def completeName(self):
        if self.parent is None:
            return self.name
        return self.parent.completeName() + '_' + self.name

    def firstParent(self):
        """
        Returns the main parent of the layer (the layer that contains
        everything)
        Actually, it's the model the user want to convert
        """
        if self.parent is None:
            return self
        return self.parent.firstParent()

    def connectionsAmongChildren(self, attr, reverse=False):
        """
        Describes connections among self's children

        Arguments:
            -attr:
                One of self.children, or 'IN', or 'OUT'
            -reverse (bool):
                If reverse == False, then look for connected children after
                attr's output
                Else, then look for connected children before attr's
                input

        Returns:
            A set containing children connected to attr, and 'IN' or 'OUT' if
            attr is connected to input or output of self

        Examples:
            If self is a Sequential and attr == 'IN' and reverse == False
            connectionsAmongChildren will return a set containing only the
            first layer in self

            If self is a Sequential and attr == 'OUT' and reverse == True
            connectionsAmongChildren will return a set containing only the
            last layer in self

            If self is a Sequential and attr is the 1st layer and
            reverse == False,
            connectionsAmongChildren will return a set containing only the
            second layer in self
        """
        connected = set()

        if isinstance(attr, str):
            if (attr != 'IN' or reverse) and not(attr == 'OUT' and reverse):
                # There's no layer which output goes to self's input, and
                # there's no layer which input comes from self's output
                return set()
            else:
                for child in self.children:
                    if attr == 'IN' and 0 in child.InputConnectedTo:
                        connected.add(child)
                    elif attr == 'OUT' and 0 in child.OutputConnectedTo:
                        connected.add(child)
                return connected
        else:
            child = attr

            if child not in self.children:
                return set()

            if not reverse:
                # not reverse : we look where child's output goes
                for bro in self.children:
                    if child is not bro and child in bro.InputConnectedTo:
                        connected.add(bro)
                for Output in child.OutputConnectedTo:
                    if not Output == 0:
                        connected.add(Output)
                    else:
                        connected.add('OUT')

            elif reverse:
                # reverse : we look where child's input comes from
                for bro in self.children:
                    if child is not bro and child in bro.OutputConnectedTo:
                        connected.add(bro)
                for Input in child.OutputConnectedTo:
                    if not Input == 0:
                        connected.add(Input)
                    else:
                        connected.add('IN')

            return connected

    def connectedChildren(self, attr, reverse=False):
        """
        Tells which simple layer(s) is connected to attr. (attr must be one of
        self's children)
        A simple layer is a layer which doesn't have any sub-layer

        Note: returned layer may not be in self.children, for example if
        self's children also have children

        Arguments:
            -attr:
                One of self.children, or 'IN', or 'OUT'
            -reverse (bool):
                If reverse == False, then look for connected layers after
                attr's output
                Else, then look for connected layers before attr's
                input

        Returns:
            A set containing ismple layers connected to attr, and 'IN' or 'OUT'
            if attr is connected to input or output of self
        """
        connected = self.connectionsAmongChildren(attr, reverse=reverse)

        connectedSimple = set()

        for layer in connected:
            if isinstance(layer, str):
                # Should be 'IN' or 'OUT'
                if self.parent is None:
                    # No parent => No brother
                    connectedSimple.add(layer)
                else:
                    # We have to look which layer is actually connected among
                    # self's brothers
                    parent = self.parent
                    cnctdRecursive = parent.connectedChildren(self,
                                                              reverse=reverse)
                    for simpleLayer in cnctdRecursive:
                        connectedSimple.add(simpleLayer)

            elif not layer.children:
                # We found a simple layer !
                connectedSimple.add(layer)

            elif layer.children:
                # Not a simple layer, we have to use recursivity
                if reverse:
                    cnctdRecursive = layer.connectedChildren('OUT',
                                                             reverse=reverse)
                else:
                    cnctdRecursive = layer.connectedChildren('IN',
                                                             reverse=reverse)
                for simpleLayer in cnctdRecursive:
                    connectedSimple.add(simpleLayer)

        return connectedSimple

    def numberOfEquivalents(self, framework=None, file=False):
        """
        Uses recursivity to compute total number of sub layers (children)
        contained in the model which have their equivalent in keras or
        pyTorch (argument framework)

        Arguments:
            -framework:
                'keras' or 'torch'
            -file:
                If true: look in equivalents AND equivalentsTxt instead of just
                equivalent

        Return:
            The number of sub layers (children) which keras or torch equivalent
            is available
        """
        number = 0
        for child in self.children:
            if not file:
                if framework in child.equivalent.keys():
                    number += 1
            elif file and framework in child.equivalentTxt.keys():
                if framework in child.equivalent.keys():
                    number += 1
            number += child.numberOfEquivalents(framework=framework)
        return number

    def childrenEquivalentsCompleted(self, framework=None, file=False):
        """
        Tells if all equivalents are available in self.children

        Arguments:
            -framework:
                'torch' or 'keras' or None. The framework to look for.
                If None, look for ANY framework
            -file:
                If true: look in equivalents AND equivalentsTxt instead of just
                equivalent

        Returns:
            True if all equivalents are available
            False if not
        """
        for child in self.children:
            if framework not in child.equivalent.keys():
                return False
            if file and framework not in child.equivalentTxt.keys():
                return False
            if (framework is None) and (child.equivalent == {}):
                return False
            if (framework is None) and file and (child.equivalentTxt == {}):
                return False
        return True

    def Connect2Layers(self, name0, name1, connectKeras=True):
        """
        Connect together 2 layers among self.children :

        Output of name0 goes to input of name1

        Arguments:
            -name0 (str)
            -name1 (str)
            -connectKeras (bool):
                If True, also connect kerasOutput to kerasInput.
        """
        child0 = self.getChild(name=name0)
        child1 = self.getChild(name=name1)
        if child0 is None or child1 is None:
            return None
        child0.OutputConnectedTo.add(child1)
        child1.InputConnectedTo.add(child0)

        if connectKeras:
            child1.kerasInput = child0.kerasOutput

    def ConnectLayers(self, *names, **kwargs):
        """
        Connect together many layers among self.children :

        Output of names[i] goes to input of names[i+1]

        Arguments:
            -*names (str):
                Children to connect together
            -connectKeras (bool):
                If True, also connect kerasOutput to kerasInput.
        """
        if 'connectKeras' in kwargs.keys():
            connectKeras = kwargs['connectKeras']
        else:
            connectKeras = True

        for i in range(len(names)-1):
            self.Connect2Layers(names[i],
                                names[i+1],
                                connectKeras=connectKeras)

    def ConnectModelInputToChildren(self, *names, **kwargs):
        """
        Puts model's input on each child given in argument

        Arguments:
            -*names (str):
                Children to connect to model's input
            -connectKeras (bool):
                If True, also connect kerasOutput to kerasInput.

        Raises:
            -ImportError:
                If keras isn't available
        """
        if 'connectKeras' in kwargs.keys():
            connectKeras = kwargs['connectKeras']
        else:
            connectKeras = True

        for name in names:
            child = self.getChild(name=name)
            if child is not None:
                child.InputConnectedTo.add(0)

                if connectKeras:
                    if self.kerasInput is None:
                        if keras is None:
                            err = "Could not import keras. Conversion failed !"
                            raise ImportError(err)
                        Input = keras.layers.Input(shape=self.input_shape)
                        self.kerasInput = Input
                    child.kerasInput = self.kerasInput

    def ConnectChildrenOutputToModel(self, *names, **kwargs):
        """
        Concatenate outputs of every child given in argument, and connect it to
        model's output

        Arguments:
            -*names (str):
                Children to connect to model's output
            -connectKeras (bool):
                If True, also connect generated kerasOutput to model's
                kerasOutput

        Raises:
            -ImportError:
                If keras isn't available
        """
        if 'connectKeras' in kwargs.keys():
            connectKeras = kwargs['connectKeras']
        else:
            connectKeras = True

        if connectKeras:
            kerasOutputs = list()

        for name in names:
            child = self.getChild(name=name)
            if child is not None:
                child.OutputConnectedTo.add(0)

                if connectKeras:
                    kerasOutputs.append(child.kerasOutput)

        if connectKeras:
            if None in kerasOutputs:
                return None
            elif len(kerasOutputs) == 0:
                return None
            elif len(kerasOutputs) == 1:
                self.kerasOutput = kerasOutputs[0]
            else:
                cat = keras.layers.concatenate(kerasOutputs, axis=1)
                self.kerasOutput = cat

    def ConnectChildrenOutputToChild(self, *names, **kwargs):
        """
        Concatenate outputs of every child given in argument and put in on
        childName's Input

        Arguments:
            -*names (str):
                Children to connect to childName's input
            -childName (str)
                A layer of the model
            -connectKeras (bool):
                If True, also connect generated kerasOutput to childName's
                kerasInput

        Raises:
            -ImportError:
                If keras isn't available
        """
        if 'connectKeras' in kwargs.keys():
            connectKeras = kwargs['connectKeras']
        else:
            connectKeras = True

        childName = kwargs['childName']

        if connectKeras:
            kerasOutputs = list()

        child = self.getChild(name=childName)
        for i in range(len(names)):
            if isinstance(names[i], str):
                child_i = self.getChild(name=names[i])
            else:
                child_i = child_i
            if child_i is not None:
                child_i.OutputConnectedTo.add(child)
                child.InputConnectedTo.add(child_i)

                if connectKeras:
                    kerasOutputs.append(child_i.kerasOutput)

        if connectKeras:
            if None in kerasOutputs:
                return None
            elif len(kerasOutputs) == 0:
                return None
            elif len(kerasOutputs) == 1:
                self.kerasOutput = kerasOutputs[0]
            else:
                cat = keras.layers.concatenate(kerasOutputs, axis=1)
                self.getChild(name=childName).kerasInput = cat

    def ConnectChildrenOutputToChildren(self, *names, **kwargs):
        """
        Concatenate outputs of every child given in argument and put in on
        childrenNames's Input

        Arguments:
            -*names (str):
                Children to connect to childrenNames's input
            -childrenNames (list)
                A list layers of the model
            -connectKeras (bool):
                If True, also connect generated kerasOutput to childrenNames's
                kerasInput

        Raises:
            -ImportError:
                If keras isn't available
        """
        if 'connectKeras' in kwargs.keys():
            connectKeras = kwargs['connectKeras']
        else:
            connectKeras = True

        childrenNames = kwargs['childrenNames']

        if isinstance(childrenNames, str):
            self.ConnectChildrenOutputToChild(*names,
                                              childName=childrenNames,
                                              connectKeras=connectKeras)
        elif isinstance(childrenNames, list):
            for child in childrenNames:
                self.ConnectChildrenOutputToChild(*names,
                                                  childName=child,
                                                  connectKeras=connectKeras)

    # --- PYTORCH SPECIFIC METHODS ---

    def isTorchBuiltIn(self):
        dT = self.detailedType
        return 'torch' in dT and 'torchvision' not in dT

    def isContainer(self):
        return ('container' in self.detailedType)

    def isTorchLayer(self):
        return self.isTorchBuiltIn() and not self.isContainer()

    def isTorchContainer(self):
        return self.isTorchBuiltIn() and self.isContainer()

    # --- KERAS SPECIFIC METHODS ---

    def kerasInputShape(self):
        """
        Determine input shape according to self.kerasInput

        Useful to check that PyTorch to Keras conversion doesn't fail
        """
        return t2k.kerasShape(self.kerasInput)

    def kerasOutputShape(self):
        """
        Determine output shape according to self.kerasOutput

        Useful to check that PyTorch to Keras conversion doesn't fail
        """
        return t2k.kerasShape(self.kerasOutput)

    # --- REPRESENTATION METHODS ---

    def DOT(self, shapes=True, debug=False):
        """
        Creates a DOT graph of the model

        Arguments:
            -shapes (bool):
                Show input and output shapes on the graph
            -debug (bool):
                Show as much information as possible

        Returns:
            A graphviz.Digraph object
            (or None if graphviz isn't available)
        """
        if graphviz is None:
            return None

        if debug:
            shapes = True

        # Step 1 : create a digraph. In most cases, names are based on id(self)
        # to ensure uniqueness
        dot = graphviz.Digraph(name='cluster_{}'.format(str(id(self))),
                               format='svg')

        # Step 2 : give our digraph a label and a color
        label = DOTlabel(model=self,
                         shapes=shapes,
                         debug=debug,
                         name=str(self))

        color = DOTcolor(model=self,
                         debug=debug)

        dot.attr(label=label, fontsize='12', color=color)

        # Step 3 : add sub layers in the digraph
        for child in self.children:

            if not child.children:
                # If there aren't any sub children
                label = DOTlabel(model=child,
                                 shapes=shapes,
                                 debug=debug,
                                 name=child.name)

                color = DOTcolor(model=child,
                                 debug=debug)

                dot.node(str(id(child)),
                         label=label,
                         color=color,
                         shape='box',
                         fontsize='11')
            else:
                # If there are sub children => recursivity
                dot.subgraph(child.DOT(shapes=shapes, debug=debug))

        # Step 4 : if it's the main layer (the whole model)
        if self.parent is None:  # Main layer (the entire model)
            Dot = graphviz.Digraph(name='all', format='svg')
            Dot.subgraph(dot)

            connectedIN = self.connectedChildren('IN')
            connectedOUT = self.connectedChildren('OUT', reverse=True)

            # Create 'IN' and 'OUT' nodes if necessary
            if shapes:
                if connectedIN:
                    Dot.node('IN', label='IN\n'+str(self.input_shape))
                if connectedOUT:
                    Dot.node('OUT', label='OUT\n'+str(self.output_shape))
            else:
                if connectedIN:
                    Dot.node('IN')
                if connectedOUT:
                    Dot.node('OUT')

            # Add edges between layers to show how they are connected
            Dot = createDOTedges(self, Dot, debug=debug)

            return Dot

        return dot

    def progression(self, framework=None):
        """
        Returns the fraction of the model (in %) which have an equivalent
        available in keras or pyTorch (argument framework)
        """
        equivalents = self.numberOfEquivalents(framework=framework)
        if framework in self.equivalent.keys():
            equivalents += 1
        total = self.numberOfChildren() + 1

        return (equivalents / total) * 100

    def summary(self, niv=0):
        """
        Prints a quick summary of the model. Actually, it's just an exhaustive
        list of layers contained in the model.

        This method doesn't tell how layers are connected to each other
        """
        if niv == 0:
            print("\nSummary of {}:\n".format(self.type))
            print('( - ): ' + self.type)
            niv = 1

        for child in self.children:
            print_str = str()
            for _ in range(niv):
                # Let's add some indentations:
                print_str = print_str + '|  '
            print_str = print_str + str(child)
            print(print_str)

            child.summary(niv+1)

    def __str__(self):
        if self.name == '':
            return self.type
        return '(' + self.name + '): ' + self.type

    def __repr__(self):
        return '<{} at {}>'.format(str(self), str(hex(id(self))))


# --- USEFUL FUNCTIONS FOR DOT CREATIONS ---

def createDOTedges(model, dot, debug=False):
    """
    Function creating edges in a DOT graph showing connections between layers

    Arguments:
        -model:
            the LayerRepresentation associated with the graph
        -dot:
            The dot graph, without edges
        -debug (bool):
            Show as muchinformation as possible

    Returns:
        A dot graph, with edges
    """
    Dot = dot.copy()

    for child in model.allChildren():
        if not child.children:
            connected = child.parent.connectedChildren(child)

            for connectedLayer in connected:
                kwargs = dict()

                if debug:
                    kwargs['label'] = str(child.kerasOutputShape())
                    kwargs['fontsize'] = '10'

                edgeBegin = str(id(child))

                if child.kerasOutput is None and debug:
                    kwargs['style'] = 'dashed'

                if connectedLayer == 'OUT':
                    edgeEnd = 'OUT'

                else:
                    edgeEnd = str(id(connectedLayer))

                    if connectedLayer.kerasInput is None and debug:
                        kwargs['style'] = 'dashed'

                Dot.edge(edgeBegin,
                         edgeEnd,
                         **kwargs)

    connectedIN = model.connectedChildren('IN')

    for layer in connectedIN:
        kwargs = dict()

        if debug:
            kwargs['label'] = str(model.kerasInputShape())
            kwargs['fontsize'] = '10'

        edgeBegin = 'IN'

        if layer == 'OUT':
            edgeEnd = 'OUT'
        else:
            edgeEnd = str(id(layer))

            if model.kerasInput is None and debug:
                kwargs['style'] = 'dashed'

        Dot.edge(edgeBegin,
                 edgeEnd,
                 **kwargs)
    return Dot


def DOTlabel(model, shapes, debug, name):
    """
    Function creating labels for dot graphs and nodes
    """
    if debug:
        # We indicate if Keras Input/Output is available
        if model.kerasInput is None:
            inputState = ' (Keras-Not Computed)'
        else:
            inputState = ' (Keras-Computed)'
        if model.kerasOutput is None:
            outputState = ' (Keras-Not Computed)'
        else:
            outputState = ' (Keras-Computed)'

        inputStr = str(model.input_shape) + inputState + '\n'
        outputStr = '\n' + str(model.output_shape) + outputState

    elif shapes:
        # We just indicate the shapes
        inputStr = str(model.input_shape) + '\n'
        outputStr = '\n' + str(model.output_shape)

    else:
        inputStr = ''
        outputStr = ''

    label = inputStr + name + outputStr
    return label


def DOTcolor(model, debug):
    """
    Function creating colors for dot graphs and nodes
    """
    if debug:
        availableEquiv = model.equivalent.keys()
        if 'keras' in availableEquiv and 'torch' in availableEquiv:
            color = 'green'
        else:
            color = 'red'
    else:
        color = 'black'

    return color
