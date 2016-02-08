'''
'''
def get_input_var(output_layer):
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        layer = layer.input_layer
    return layer.input_var
