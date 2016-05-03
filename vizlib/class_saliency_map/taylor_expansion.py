'''Implement class saliency visualization as per [1]

The key idea here is that we want to rank the pixels of an image based on their
influence on the class scores.

If the class scores were a linear function of the input pixels this would be
easy:

    S_c(I) = w_{c}^{t}I + b_c

The weights would indicate how each pixel should be ranked.

We can create a locally linear approximation of S_c using taylor series
expansion, by defining w as:

    w = dS_c/dI | I_0

Here I_0 is the image to evaluate at.

[1] simonyan2013visualizing.pdf
'''
import vizlib
import theano

def taylor_expansion_functions(output_layer, ignore_nonlinearity=True):
    X = vizlib.utils.get_input_var(output_layer)
    scores = vizlib.activation_maximization.scores(
        X, output_layer, ignore_nonlinearity=ignore_nonlinearity)
    return [
        theano.function([X], theano.grad(score, wrt=X).max(axis=1)[0])
        for score in scores
    ]

def taylor_expansion(X, output_layer, output_node, ignore_nonlinearity=True):
    fs = taylor_expansion_functions(output_layer, ignore_nonlinearity)
    return fs[output_node](X)
