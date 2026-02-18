#include "layer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

dense_layer_t* layer_create(size_t input_size, size_t output_size, activation_type_t activation) {
    dense_layer_t *layer = malloc(sizeof(dense_layer_t));
    if (!layer) return NULL;

    layer->weights = tensor_create(input_size, output_size);
    layer->bias = tensor_create(1, output_size);

    layer->grad_weights = tensor_create(input_size, output_size);
    layer->grad_bias = tensor_create(1, output_size);

    if (!layer->weights || !layer->bias || 
        !layer->grad_weights || !layer->grad_bias) {
        layer_destroy(layer);
        return NULL;
    }

    layer->input = NULL;
    layer->output = NULL;
    layer->pre_activation = NULL;

    layer->activation = activation;

    layer_xavier_init(layer);
    tensor_zeros(layer->bias);

    return layer;
}

void layer_destroy(dense_layer_t *layer) {
    if (!layer) return;

    tensor_destroy(layer->weights);
    tensor_destroy(layer->bias);
    tensor_destroy(layer->input);
    tensor_destroy(layer->output);
    tensor_destroy(layer->pre_activation);
    tensor_destroy(layer->grad_weights);
    tensor_destroy(layer->grad_bias);

    free(layer);
}
void layer_xavier_init(dense_layer_t *layer) {
    float limit = sqrtf(6.0f /
        (layer->weights->rows + layer->weights->cols));
    tensor_random(layer->weights, -limit, limit);
}

void layer_he_init(dense_layer_t *layer) {
 
    float limit = sqrtf(6.0f / layer->weights->rows);
    tensor_random(layer->weights, -limit, limit);
}

tensor_t* layer_forward(dense_layer_t *layer, const tensor_t *input) {

    if (layer->input) tensor_destroy(layer->input);
    layer->input = tensor_copy(input);

    tensor_t *matmul_result = tensor_matmul(input, layer->weights);
    if (!matmul_result) return NULL;

    if (layer->pre_activation) tensor_destroy(layer->pre_activation);
    layer->pre_activation = tensor_create(
        matmul_result->rows, matmul_result->cols);
    for (size_t i = 0; i < matmul_result->rows; i++) {
        for (size_t j = 0; j < matmul_result->cols; j++) {
            layer->pre_activation->data[i * matmul_result->cols + j] =
                matmul_result->data[i * matmul_result->cols + j] +
                layer->bias->data[j];
        }
    }

    tensor_destroy(matmul_result);

    if (layer->output) tensor_destroy(layer->output);
    layer->output = tensor_create(
        layer->pre_activation->rows,
        layer->pre_activation->cols);

    switch (layer->activation) {
        case ACTIVATION_RELU:
            tensor_relu(layer->pre_activation, layer->output);
            break;

        case ACTIVATION_SIGMOID:
            tensor_sigmoid(layer->pre_activation, layer->output);
            break;

        case ACTIVATION_NONE:
        default:
            tensor_copy_data(layer->output,
                             layer->pre_activation);
            break;
    }

    return layer->output;
}

tensor_t* layer_backward(dense_layer_t *layer,
                         const tensor_t *grad_output) {

    size_t batch_size = grad_output->rows;

    tensor_t *grad_activation =
        tensor_create(grad_output->rows,
                      grad_output->cols);

    switch (layer->activation) {

        case ACTIVATION_RELU: {
            tensor_t *relu_deriv =
                tensor_create(layer->pre_activation->rows,
                              layer->pre_activation->cols);

            tensor_relu_derivative(layer->pre_activation,
                                   relu_deriv);

            tensor_multiply(grad_output,
                            relu_deriv,
                            grad_activation);

            tensor_destroy(relu_deriv);
            break;
        }

        case ACTIVATION_SIGMOID: {
            tensor_t *sigmoid_deriv =
                tensor_create(layer->output->rows,
                              layer->output->cols);

            tensor_sigmoid_derivative(layer->output,
                                      sigmoid_deriv);

            tensor_multiply(grad_output,
                            sigmoid_deriv,
                            grad_activation);

            tensor_destroy(sigmoid_deriv);
            break;
        }

        case ACTIVATION_NONE:
        default:
            tensor_copy_data(grad_activation, grad_output);
            break;
    }

    tensor_t *input_T = tensor_transpose(layer->input);
    tensor_t *grad_w = tensor_matmul(input_T, grad_activation);


    for (size_t i = 0;
         i < grad_w->rows * grad_w->cols; i++) {
        grad_w->data[i] /= batch_size;
    }

    tensor_copy_data(layer->grad_weights, grad_w);

    tensor_destroy(input_T);
    tensor_destroy(grad_w);
    tensor_zeros(layer->grad_bias);

    for (size_t i = 0; i < grad_activation->rows; i++) {
        for (size_t j = 0; j < grad_activation->cols; j++) {
            layer->grad_bias->data[j] +=
                grad_activation->data[i * grad_activation->cols + j];
        }
    }

    for (size_t j = 0; j < grad_activation->cols; j++) {
        layer->grad_bias->data[j] /= batch_size;
    }

    tensor_t *weights_T = tensor_transpose(layer->weights);
    tensor_t *grad_input =
        tensor_matmul(grad_activation, weights_T);

    tensor_destroy(weights_T);
    tensor_destroy(grad_activation);

    return grad_input;
}

size_t layer_param_count(const dense_layer_t *layer) {
    return (layer->weights->rows *
            layer->weights->cols)
            + layer->bias->cols;
}
