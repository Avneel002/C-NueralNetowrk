#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} tensor_t;

tensor_t* tensor_create(size_t rows, size_t cols);
void tensor_destroy(tensor_t *tensor);

void tensor_fill(tensor_t *tensor, float value);
void tensor_random(tensor_t *tensor, float min, float max);
void tensor_zeros(tensor_t *tensor);
void tensor_ones(tensor_t *tensor);

void tensor_print(const tensor_t *tensor, const char *name);
tensor_t* tensor_copy(const tensor_t *src);
void tensor_copy_data(tensor_t *dst, const tensor_t *src);

void tensor_add(const tensor_t *a, const tensor_t *b, tensor_t *result);
void tensor_subtract(const tensor_t *a, const tensor_t *b, tensor_t *result);
void tensor_multiply(const tensor_t *a, const tensor_t *b, tensor_t *result);
void tensor_scale(tensor_t *tensor, float scalar);

tensor_t* tensor_matmul(const tensor_t *a, const tensor_t *b);
tensor_t* tensor_transpose(const tensor_t *tensor);


void tensor_relu(const tensor_t *input, tensor_t *output);
void tensor_relu_derivative(const tensor_t *input, tensor_t *output);
void tensor_sigmoid(const tensor_t *input, tensor_t *output);
void tensor_sigmoid_derivative(const tensor_t *input, tensor_t *output);

#endif