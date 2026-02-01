#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

tensor_t* tensor_create(size_t rows, size_t cols) {
    tensor_t *tensor = (tensor_t*)malloc(sizeof(tensor_t));
    if (!tensor) {
        fprintf(stderr, "Failed to allocate tensor structure\n");
        return NULL;
    }
    
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->data = (float*)calloc(rows * cols, sizeof(float));
    
    if (!tensor->data) {
        fprintf(stderr, "Failed to allocate tensor data\n");
        free(tensor);
        return NULL;
    }
    
    return tensor;
}

void tensor_destroy(tensor_t *tensor) {
    if (tensor) {
        if (tensor->data) {
            free(tensor->data);
        }
        free(tensor);
    }
}

void tensor_fill(tensor_t *tensor, float value) {
    for (size_t i = 0; i < tensor->rows * tensor->cols; i++) {
        tensor->data[i] = value;
    }
}

void tensor_random(tensor_t *tensor, float min, float max) {
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    for (size_t i = 0; i < tensor->rows * tensor->cols; i++) {
        float random = (float)rand() / RAND_MAX;
        tensor->data[i] = min + random * (max - min);
    }
}

void tensor_zeros(tensor_t *tensor) {
    tensor_fill(tensor, 0.0f);
}

void tensor_ones(tensor_t *tensor) {
    tensor_fill(tensor, 1.0f);
}

void tensor_print(const tensor_t *tensor, const char *name) {
    printf("%s (%zu x %zu):\n", name, tensor->rows, tensor->cols);
    for (size_t i = 0; i < tensor->rows; i++) {
        for (size_t j = 0; j < tensor->cols; j++) {
            printf("%8.4f ", tensor->data[i * tensor->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

tensor_t* tensor_copy(const tensor_t *src) {
    tensor_t *dst = tensor_create(src->rows, src->cols);
    if (dst) {
        memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
    }
    return dst;
}

void tensor_copy_data(tensor_t *dst, const tensor_t *src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        fprintf(stderr, "Tensor dimensions don't match for copy\n");
        return;
    }
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

void tensor_add(const tensor_t *a, const tensor_t *b, tensor_t *result) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Tensor dimensions don't match for addition\n");
        return;
    }
    
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_subtract(const tensor_t *a, const tensor_t *b, tensor_t *result) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Tensor dimensions don't match for subtraction\n");
        return;
    }
    
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

void tensor_multiply(const tensor_t *a, const tensor_t *b, tensor_t *result) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Tensor dimensions don't match for multiplication\n");
        return;
    }
    
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_scale(tensor_t *tensor, float scalar) {
    for (size_t i = 0; i < tensor->rows * tensor->cols; i++) {
        tensor->data[i] *= scalar;
    }
}

tensor_t* tensor_matmul(const tensor_t *a, const tensor_t *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Invalid dimensions for matrix multiplication\n");
        return NULL;
    }
    
    tensor_t *result = tensor_create(a->rows, b->cols);
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
    
    return result;
}

tensor_t* tensor_transpose(const tensor_t *tensor) {
    tensor_t *result = tensor_create(tensor->cols, tensor->rows);
    if (!result) return NULL;
    
    for (size_t i = 0; i < tensor->rows; i++) {
        for (size_t j = 0; j < tensor->cols; j++) {
            result->data[j * result->cols + i] = tensor->data[i * tensor->cols + j];
        }
    }
    
    return result;
}

void tensor_relu(const tensor_t *input, tensor_t *output) {
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        output->data[i] = fmaxf(0.0f, input->data[i]);
    }
}

void tensor_relu_derivative(const tensor_t *input, tensor_t *output) {
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        output->data[i] = input->data[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void tensor_sigmoid(const tensor_t *input, tensor_t *output) {
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
    }
}

void tensor_sigmoid_derivative(const tensor_t *input, tensor_t *output) {
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        float s = input->data[i];
        output->data[i] = s * (1.0f - s);
    }
}