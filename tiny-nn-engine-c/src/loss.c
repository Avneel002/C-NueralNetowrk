#include "loss.h"
#include <math.h>
#include <stdio.h>

static int check_dimensions(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) {
        fprintf(stderr, "Null tensor pointer\n");
        return 0;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Dimension mismatch\n");
        return 0;
    }
    if (!a->data || !b->data) {
        fprintf(stderr, "Tensor data is NULL\n");
        return 0;
    }
    return 1;
}

float loss_mse(const tensor_t *predictions, const tensor_t *targets) {
    if (!check_dimensions(predictions, targets))
        return 0.0f;

    size_t n = predictions->rows * predictions->cols;
    if (n == 0)
        return 0.0f;

    float sum = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float diff = predictions->data[i] - targets->data[i];
        sum += diff * diff;
    }

    return sum / (float)n;
}

float loss_binary_crossentropy(const tensor_t *predictions, const tensor_t *targets) {
    if (!check_dimensions(predictions, targets))
        return 0.0f;

    size_t n = predictions->rows * predictions->cols;
    if (n == 0)
        return 0.0f;

    const float epsilon = 1e-7f;
    float sum = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float p = predictions->data[i];

        if (p < epsilon) p = epsilon;
        if (p > 1.0f - epsilon) p = 1.0f - epsilon;

        float t = targets->data[i];

        sum += -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }

    return sum / (float)n;
}

void loss_mse_derivative(const tensor_t *predictions,
                         const tensor_t *targets,
                         tensor_t *grad) {

    if (!check_dimensions(predictions, targets) ||
        !check_dimensions(predictions, grad))
        return;

    size_t n = predictions->rows * predictions->cols;
    if (n == 0)
        return;

    for (size_t i = 0; i < n; i++) {
        grad->data[i] = 2.0f * (predictions->data[i] - targets->data[i]) / (float)n;
    }
}

void loss_bce_derivative(const tensor_t *predictions,
                         const tensor_t *targets,
                         tensor_t *grad) {

    if (!check_dimensions(predictions, targets) ||
        !check_dimensions(predictions, grad))
        return;

    size_t n = predictions->rows * predictions->cols;
    if (n == 0)
        return;

    const float epsilon = 1e-7f;

    for (size_t i = 0; i < n; i++) {
        float p = predictions->data[i];

        
        if (p < epsilon) p = epsilon;
        if (p > 1.0f - epsilon) p = 1.0f - epsilon;

        float t = targets->data[i];

        grad->data[i] =
            (p - t) / ((p * (1.0f - p)) * (float)n);
    }
}
