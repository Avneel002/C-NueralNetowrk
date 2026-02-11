#include "optimizer.h"
#include <stdlib.h>
#include <math.h>


sgd_optimizer_t* sgd_create(float learning_rate) {
    sgd_optimizer_t *opt = (sgd_optimizer_t*)malloc(sizeof(sgd_optimizer_t));
    opt->learning_rate = learning_rate;
    return opt;
}

void sgd_step(sgd_optimizer_t *opt, dense_layer_t *layer) {
    for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        layer->weights->data[i] -= opt->learning_rate * layer->grad_weights->data[i];
    }
    
    
    for (size_t i = 0; i < layer->bias->cols; i++) {
        layer->bias->data[i] -= opt->learning_rate * layer->grad_bias->data[i];
    }
}

void sgd_destroy(sgd_optimizer_t *opt) {
    free(opt);
}

adam_optimizer_t* adam_create(float learning_rate, size_t num_layers) {
    adam_optimizer_t *opt = (adam_optimizer_t*)malloc(sizeof(adam_optimizer_t));
    
    opt->learning_rate = learning_rate;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;
    opt->timestep = 0;
    opt->num_layers = num_layers;
    
    opt->m_weights = (tensor_t**)calloc(num_layers, sizeof(tensor_t*));
    opt->v_weights = (tensor_t**)calloc(num_layers, sizeof(tensor_t*));
    opt->m_bias = (tensor_t**)calloc(num_layers, sizeof(tensor_t*));
    opt->v_bias = (tensor_t**)calloc(num_layers, sizeof(tensor_t*));
    
    return opt;
}

void adam_step(adam_optimizer_t *opt, dense_layer_t **layers, size_t num_layers) {
    opt->timestep++;
    
    float lr_t = opt->learning_rate * sqrtf(1.0f - powf(opt->beta2, opt->timestep)) 
                  / (1.0f - powf(opt->beta1, opt->timestep));
    
    for (size_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        dense_layer_t *layer = layers[layer_idx];
        
        
        if (opt->m_weights[layer_idx] == NULL) {
            opt->m_weights[layer_idx] = tensor_create(layer->weights->rows, layer->weights->cols);
            opt->v_weights[layer_idx] = tensor_create(layer->weights->rows, layer->weights->cols);
            opt->m_bias[layer_idx] = tensor_create(1, layer->bias->cols);
            opt->v_bias[layer_idx] = tensor_create(1, layer->bias->cols);
            
            tensor_zeros(opt->m_weights[layer_idx]);
            tensor_zeros(opt->v_weights[layer_idx]);
            tensor_zeros(opt->m_bias[layer_idx]);
            tensor_zeros(opt->v_bias[layer_idx]);
        }
        
        
        for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
            float g = layer->grad_weights->data[i];
            
            
            opt->m_weights[layer_idx]->data[i] = opt->beta1 * opt->m_weights[layer_idx]->data[i] 
                                                  + (1.0f - opt->beta1) * g;
            
            opt->v_weights[layer_idx]->data[i] = opt->beta2 * opt->v_weights[layer_idx]->data[i] 
                                                  + (1.0f - opt->beta2) * g * g;
            
            layer->weights->data[i] -= lr_t * opt->m_weights[layer_idx]->data[i] 
                                        / (sqrtf(opt->v_weights[layer_idx]->data[i]) + opt->epsilon);
        }
        for (size_t i = 0; i < layer->bias->cols; i++) {
            float g = layer->grad_bias->data[i];
            
            opt->m_bias[layer_idx]->data[i] = opt->beta1 * opt->m_bias[layer_idx]->data[i] 
                                               + (1.0f - opt->beta1) * g;
            
            opt->v_bias[layer_idx]->data[i] = opt->beta2 * opt->v_bias[layer_idx]->data[i] 
                                               + (1.0f - opt->beta2) * g * g;
            
            layer->bias->data[i] -= lr_t * opt->m_bias[layer_idx]->data[i] 
                                     / (sqrtf(opt->v_bias[layer_idx]->data[i]) + opt->epsilon);
        }
    }
}

void adam_destroy(adam_optimizer_t *opt) {
    for (size_t i = 0; i < opt->num_layers; i++) {
        if (opt->m_weights[i]) tensor_destroy(opt->m_weights[i]);
        if (opt->v_weights[i]) tensor_destroy(opt->v_weights[i]);
        if (opt->m_bias[i]) tensor_destroy(opt->m_bias[i]);
        if (opt->v_bias[i]) tensor_destroy(opt->v_bias[i]);
    }
    
    free(opt->m_weights);
    free(opt->v_weights);
    free(opt->m_bias);
    free(opt->v_bias);
    free(opt);
}