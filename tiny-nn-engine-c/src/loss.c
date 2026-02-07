#include "loss.h"
#include <math.h>
#include <stdio.h>

float loss_mse(const tensor_t *predictions, const tensor_t *targets) {
    if (predictions->rows != targets->rows || predictions->cols != targets->cols) {
        fprintf(stderr, "Dimension mismatch in MSE loss\n");
        return 0.0f;
    }
    
    float sum = 0.0f;
    size_t n = predictions->rows * predictions->cols;
    
    for (size_t i = 0; i < n; i++) {
        float diff = predictions->data[i] - targets->data[i];
        sum += diff * diff;
    }
    
    return sum / n;
}

float loss_binary_crossentropy(const tensor_t *predictions, const tensor_t *targets) {
    if (predictions->rows != targets->rows || predictions->cols != targets->cols) {
        fprintf(stderr, "Dimension mismatch in BCE loss\n");
        return 0.0f;
    }
    
    float sum = 0.0f;
    size_t n = predictions->rows * predictions->cols;
    const float epsilon = 1e-7f;
    
    for (size_t i = 0; i < n; i++) {
        float p = fmaxf(fminf(predictions->data[i], 1.0f - epsilon), epsilon);
        float t = targets->data[i];
        sum += -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }
    
    return sum / n;
}

void loss_mse_derivative(const tensor_t *predictions, const tensor_t *targets, tensor_t *grad) {
    size_t n = predictions->rows * predictions->cols;
    
    for (size_t i = 0; i < n; i++) {
        grad->data[i] = 2.0f * (predictions->data[i] - targets->data[i]) / n;
    }
}

void loss_bce_derivative(const tensor_t *predictions, const tensor_t *targets, tensor_t *grad) {
    size_t n = predictions->rows * predictions->cols;
    const float epsilon = 1e-7f;
    
    for (size_t i = 0; i < n; i++) {
        float p = fmaxf(fminf(predictions->data[i], 1.0f - epsilon), epsilon);
        float t = targets->data[i];
        grad->data[i] = (p - t) / (p * (1.0f - p) * n);
    }
}