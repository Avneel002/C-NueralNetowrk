#ifndef LOSS_H
#define LOSS_H
#include "tensor.h"

float loss_mse(const tensor_t *predictions, const tensor_t *targets);
float loss_binary_crossentropy(const tensor_t *predictions, const tensor_t *targets);
void loss_mse_derivative(const tensor_t *predictions, const tensor_t *targets, tensor_t *grad);
void loss_bce_derivative(const tensor_t *predictions, const tensor_t *targets, tensor_t *grad);
#endif