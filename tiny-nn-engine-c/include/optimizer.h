#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "layer.h"

typedef struct {
    float learning_rate;
} sgd_optimizer_t;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int timestep;
    tensor_t **m_weights;
    tensor_t **v_weights;
    tensor_t **m_bias;
    tensor_t **v_bias;
    
    size_t num_layers;
} adam_optimizer_t;


sgd_optimizer_t* sgd_create(float learning_rate);
void sgd_step(sgd_optimizer_t *opt, dense_layer_t *layer);
void sgd_destroy(sgd_optimizer_t *opt);

adam_optimizer_t* adam_create(float learning_rate, size_t num_layers);
void adam_step(adam_optimizer_t *opt, dense_layer_t **layers, size_t num_layers);
void adam_destroy(adam_optimizer_t *opt);

#endif