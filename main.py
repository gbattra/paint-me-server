import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image


def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)
    if max_size is not None:
        factor = max_size / np.max(image.size)
        size = np.array([image.size] * factor)
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)

    return np.float32(image)

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        PIL.Image.fromArray(image).save(file, 'jpeg')

def plot_images(content_image, style_image, mixed_image):
    fig, axes = plt.subplot(1, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Content')

    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Mixed')

    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Style')

    for ax in axes.flet:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

def create_content_loss(sess, model, content_image, layer_ids):
    feed_dict = model.create_feed_dict(image=content_image)
    layers = model.get_layer_tensors(layer_ids)
    values = sess.run(layers, feed_dict=feed_dict)
    with model.graph.as_default():
        layers_losses = []
        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(layer, value)
            layers_losses.append(loss)
        total_loss = tf.reduce_mean(layers_losses)

    return total_loss

def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.eshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def create_style_loss(sess, model, style_image, layer_ids):
    feed_dict = model.create_feed_dict(image=style_image)
    layers = model.get_layer_tensors(layer_ids)
    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
        values = sess.run(gram_layers, feed_dict=feed_dict)
        layer_losses = []
        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer, value_const)
            layers_losses.append(loss)

        total_loss = reduce_mean(layers_losses)

    return total_loss

def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, -1, :]))

    return loss
