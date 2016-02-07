import lasagne

def expression_to_maximize(output_layer, ignore_nonlinearity=False, output_node=None):
    '''
    output_layer:        lasagne.Layer
    ignore_nonlinearity: bool
    output_node:         uint
    '''
    if ignore_nonlinearity:
        original_nonlinearity = output_layer.nonlinearity
        output_layer.nonlinearity = lasagne.nonlinearities.identity

    output = get_output_expression(output_layer)

    if ignore_nonlinearity:
        output_layer.nonlinearity = original_nonlinearity

    if output_node:
        return output[output_node]
    return output

def get_output_expression(output_layer):
    output_expr = get_output_expressions(output_layer)[-1]
    # This output is always of the form (n_batch, ...)
    # Since optimizing for a batch larger than 1 is hard to define,
    # we won't do that here. And simply return the first element in the batch.
    output_expr = output_expr[0]

    if hasattr(output_layer, 'num_units'):
        return [output_expr[i] for i in range(output_layer.num_units)]
    if hasattr(output_layer, 'num_filters'):
        output_expr = get_output_expression_conv_layer(output_expr)
        return [output_expr[i] for i in range(output_layer.num_filters)]
    raise NotImplemented('Can not get output expression for %s' % output_layer)

def get_output_expression_conv_layer(output_expr):
    # output_expr is of the shape:
    #   num_filters, width, height
    # since we can not maximize multiple outputs at the same time,
    # let's just sum the output over the right axis.
    return output_expr.sum(axis=(-2, -1))

def get_output_expressions(output_layer):
    '''Return a list of the output expressions wrt the input variable
    for all layers that precide the output_layer and the output_layer itself.

    i.e., if output_layer has 9 layers before it,
    then this will return a list x with len(x) == 10,
    where x[-1] is the output expression of the output_layer
    '''

    # We assume layers have a single input and a single output,
    # so building the computation graph is as simple as traversing
    # to the input, and then traversing back up.
    first_layer = output_layer
    layers = [output_layer]
    while not hasattr(first_layer, 'input_var'):
        first_layer = first_layer.input_layer
        layers.append(first_layer)
    # the pop removes the first_layer
    output_expr = layers.pop().input_var
    layers.reverse()
    expressions = [output_expr]

    for current_layer in layers:
        output_expr = current_layer.get_output_for(output_expr)
        expressions.append(output_expr)

    return expressions
