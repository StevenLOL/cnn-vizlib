import lasagne
import theano
import numpy as np


def maximize_scores(
    output_layer,
    X_init,
    number_of_iterations,
    ignore_nonlinearity=True,
    output_node=None,
    learning_rate=0.01,
    momentum=0.9,
    max_norm=None,
):
    '''Returns a list of (score_i, maximizer_i) where `maximizer_i` is the
    input that maximizes the activation of the `i`-th node in the `output_layer`,
    and `score_i` is the corresponding score.

    Gradient descent is used to find the maximizers.

    Use `output_node=<i>` to get the output only for the `i`-th node.

    We define the activation as the sum of the inputs, unless
    `ignore_nonlinearity` is False, then we take it to mean the output.

    Using nonlinearities in a softmax layer is potentially misleading,
    since inputs can be produced that minimize other class scores,
    rather than maximizing their own.
    '''
    X = theano.shared(value=X_init, name='X', borrow=False)
    expressions = scores(X, output_layer, ignore_nonlinearity)
    if output_node:
        expressions = expressions[output_node:output_node+1]

    results = []
    histories = []
    for expr in expressions:
        X.set_value(X_init)

        cost = -expr

        updates = lasagne.updates.nesterov_momentum(
            cost, [X],
            learning_rate=learning_rate,
            momentum=momentum
        )

        if max_norm:
            updates[X] = lasagne.updates.norm_constraint(updates[X], max_norm)

        update_iter = theano.function(
            [], cost,
            updates=updates
        )

        # lasagne expects minimization, we want maximization.
        # this is why I am messing around w/ minus signs.
        current_value = best_value = -float(update_iter())
        history = [current_value]
        best_X = X.get_value(borrow=False)

        for i in range(number_of_iterations):
            current_value = -float(update_iter())
            history.append(current_value)
            if i % 100 == 0:
                current_X = X.get_value(borrow=False)
                # We can stop early. The score can be increased indefinately
                # when we do not use a maxnorm. However, the image does not
                # fundamentally change.
                current_X_normalized = current_X / current_X.sum()
                best_X_normalized = best_X / best_X.sum()
                if np.abs(current_X_normalized - best_X_normalized).max() < 1e-6:
                    print('Early stopping')
                    break

            if current_value > best_value:
                best_X = X.get_value(borrow=False)
                best_value = current_value

        results.append((-best_value, best_X))
        histories.append(history)
    return (histories, results)

def scores(X, output_layer, ignore_nonlinearity=True):
    if ignore_nonlinearity:
        original_nonlinearity = output_layer.nonlinearity
        output_layer.nonlinearity = lasagne.nonlinearities.identity

    output = get_output_expression(X, output_layer)

    if ignore_nonlinearity:
        output_layer.nonlinearity = original_nonlinearity

    return output

def get_output_expression(X, output_layer):
    output_expr = get_output_expressions(X, output_layer)[-1]
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

def get_output_expressions(X, output_layer):
    '''Return a list of the output expressions wrt the input variable X
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
    layers.pop()
    output_expr = X
    layers.reverse()
    expressions = [output_expr]

    for current_layer in layers:
        output_expr = current_layer.get_output_for(output_expr, deterministic=True)
        expressions.append(output_expr)

    return expressions
