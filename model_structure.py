import graphviz
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from utils import build_model
from utils import arg

def draw_model(args):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(args.embedding_size, activation=tf.keras.activations.relu,
                              batch_input_shape=[args.batch_size, 30, 2]),
        tf.keras.layers.LSTM(args.rnn_size,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(5)
    ])

    plot_model(model, to_file='model.png', show_shapes=True)

if __name__ == '__main__':
    args = arg()
    draw_model(args)





"""
def draw_model(model):
    dot = graphviz.Digraph()
    dot.node('input', label='Input\n[batch_size, timesteps, 2]')
    for i, layer in enumerate(model.layers):
        name = layer.name
        label = f'{name}\n({layer.__class__.__name__})'
        dot.node(name, label=label)
        if i == 0:
            dot.edge('input', name)
        else:
            dot.edge(model.layers[i-1].name, name)
    dot.node('output', label=f'Output\n[{model.output_shape[1]}, {model.output_shape[2]}]')
    dot.edge(model.layers[-1].name, 'output')
    dot.graph_attr['rankdir'] = 'LR'
    dot.render('model', format='png', cleanup=True)
    plot_model(model, show_shapes=True, to_file='model.png')

if __name__ == '__main__':
    args = arg()
    model = build_model(args)
    draw_model(model)
"""




