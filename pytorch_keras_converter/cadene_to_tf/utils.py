
def removeBorderSpaces(inputStr):
    """
    Function that removes trailing and leading whitespaces

    Argument:
        -inputStr (str)

    Returns:
        A str of inputStr without leading and trailing whitespaces

    """
    if len(inputStr) == 0:
        return str()
    if inputStr[0] == ' ':
        return removeBorderSpaces(inputStr[1:])
    if inputStr[-1] == ' ':
        return removeBorderSpaces(inputStr[:-1])
    return inputStr


def extractFunctionArguments(fct):
    """extractFunctionArguments(fct)

    Argument:
        -fct:
            a str of the syntax used to call the function
            example: fct(a=1,b=2,c=3,4)

    Return:
        a dictionnary containing as values the arguments,
        and as keys the names of arguments.
        Unnamed arguments are keyed with their number

    Example :
        extractFunctionArguments(fct(a=1,b=2,c=3,4))
        returns
        [fct, {'a':1, 'b':2, 'c':3, 4:4}]
    """

    def searchValidElement(inputStr, element):
        # We can't use fct.split(',') because there may be some commas (',')
        # in str or other called functions
        # We'll use a for loop.

        # List that will contain the Posi of each commas
        commaPosi = list()
        # StrType1: 'str'
        # StrType2: "str"
        # StrType3: '''str'''
        # StrType6: """str"""
        inStrType1 = False
        inStrType2 = False
        inStrType3 = False
        inStrType6 = False

        openedParenthesis = 0

        for char in range(len(inputStr)):
            inStr = inStrType1 * inStrType2 * inStrType3 * inStrType6
            if inStr:
                # We are currently in a str
                if (inStrType1 and inputStr[char] == "'"):
                    inStrType1 = False
                elif (inStrType2 and inputStr[char] == '"'):
                    inStrType2 = False
                elif (inStrType3 and inputStr[char] == "'" and
                      inputStr[char-1] == "'" and inputStr[char-2] == "'"):
                    inStrType3 = False
                elif (inStrType6 and inputStr[char] == '"' and
                      inputStr[char-1] == '"' and inputStr[char-2] == '"'):
                    inStrType6 = False
            else:
                # We are *not* in a str
                if inputStr[char] == '(':
                    openedParenthesis += 1
                elif inputStr[char] == ')':
                    openedParenthesis -= 1
                elif (inputStr[char] == '"'):
                    # Starting a str of type 6 or 2
                    if char+2 < len(inputStr)-1:
                        if (inputStr[char+1] == '"' and fct[char+2] == '"'):
                            inStrType6 = True
                        else:
                            inStrType2 = True
                elif (inputStr[char] == "'"):
                    # Starting a str of type 1 or 3
                    if char+2 < len(inputStr)-1:
                        if inputStr[char+1] == "'" and inputStr[char+2] == "'":
                            inStrType3 = True
                        else:
                            inStrType1 = True
                elif inputStr[char] == element and openedParenthesis == 0:
                    # We found a valid comma !
                    commaPosi.append(char)
        return commaPosi

    # Looking for the first '('
    for char in range(len(fct)):
        if fct[char] == '(':
            start = char
            break

    # Looking for the last ')'
    for char in range(len(fct)-1, 0, -1):
        if fct[char] == ')':
            stop = char
            break

    try:
        if stop:
            pass
    except NameError:
        stop = -1

    calledFct = fct[:start]
    fct = fct[start+1:stop]
    # Now we only have what was in the parenthesis

    # Let's extract the arguments.

    commaPosi = searchValidElement(fct, ',')

    # List of arguments:
    args = list()

    for commaNumber in range(len(commaPosi)):
        if commaNumber == 0:
            args.append(fct[:commaPosi[commaNumber]])
        else:
            args.append(fct[commaPosi[commaNumber-1]+1:commaPosi[commaNumber]])

    # Don't forget the last argument !
    if commaPosi:
        args.append(fct[commaPosi[-1]+1:])
    else:
        args.append(fct)

    # Now args is a list of all arguments passed to the function
    # Example: ['arg1', 'fct(fa1,fa2)', 'arg3 = "12"']

    # We still need to analyse it to separate arguments' names and their values
    # We have to find chars "=" :

    equalsPosi = list()

    for argument in args:
        equalsPosi.append(searchValidElement(argument, '='))
        # equalsPosi[argument] should be an empty list or a list containing
        # only one integer

    # Finally, let's build the output dict
    output = dict()

    for argNumber in range(len(args)):
        if equalsPosi[argNumber]:
            name = removeBorderSpaces(
                    args[argNumber][:equalsPosi[argNumber][0]])
            value = removeBorderSpaces(
                    args[argNumber][equalsPosi[argNumber][0]+1:])
            output[name] = value
        else:
            output[argNumber] = removeBorderSpaces(args[argNumber])

    for key in output.keys():
        try:
            output[key] = int(output[key])
        except ValueError:
            pass

        if output[key] == 'True':
            output[key] = True
        if output[key] == 'False':
            output[key] = False
        if output[key] == 'None':
            output[key] = None

    return [calledFct, output]
