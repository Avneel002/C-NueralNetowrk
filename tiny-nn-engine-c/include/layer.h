#ifndef LAYER_H
#define LAYER_H
#include "tensor.h"

typedef enum {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID
} activation_type_t;

typedef struct {
    tensor_t *weights;
    tensor_t *bias;
    tensor_t *input;
    tensor_t *output;
    tensor_t *pre_activation;
    tensor_t *grad_weights;
    tensor_t *grad_bias;
    
    activation_type_t activation;
} dense_layer_t;

dense_layer_t* layer_create(size_t input_size, size_t output_size, activation_type_t activation);
void layer_destroy(dense_layer_t *layer);

void layer_xavier_init(dense_layer_t *layer);
void layer_he_init(dense_layer_t *layer);

tensor_t* layer_forward(dense_layer_t *layer, const tensor_t *input);

tensor_t* layer_backward(dense_layer_t *layer, const tensor_t *grad_output);

size_t layer_param_count(const dense_layer_t *layer);

#endif