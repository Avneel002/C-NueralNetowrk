#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/tensor.h"

#define EPSILON 1e-5f

void test_tensor_creation() {
    printf("Testing tensor creation... ");
    tensor_t *t = tensor_create(3, 4);
    assert(t != NULL);
    assert(t->rows == 3);
    assert(t->cols == 4);
    tensor_destroy(t);
    printf("✓\n");
}

void test_tensor_fill() {
    printf("Testing tensor fill... ");
    tensor_t *t = tensor_create(2, 2);
    tensor_fill(t, 5.0f);
    
    for (size_t i = 0; i < 4; i++) {
        assert(fabsf(t->data[i] - 5.0f) < EPSILON);
    }
    
    tensor_destroy(t);
    printf("✓\n");
}

void test_tensor_add() {
    printf("Testing tensor addition... ");
    tensor_t *a = tensor_create(2, 2);
    tensor_t *b = tensor_create(2, 2);
    tensor_t *result = tensor_create(2, 2);
    
    tensor_fill(a, 1.0f);
    tensor_fill(b, 2.0f);
    tensor_add(a, b, result);
    
    for (size_t i = 0; i < 4; i++) {
        assert(fabsf(result->data[i] - 3.0f) < EPSILON);
    }
    
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
    printf("✓\n");
}

void test_tensor_matmul() {
    printf("Testing matrix multiplication... ");
    tensor_t *a = tensor_create(2, 3);
    tensor_t *b = tensor_create(3, 2);
    float a_data[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++) a->data[i] = a_data[i];
    float b_data[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++) b->data[i] = b_data[i];
    
    tensor_t *result = tensor_matmul(a, b);
    float expected[] = {22, 28, 49, 64};
    for (int i = 0; i < 4; i++) {
        assert(fabsf(result->data[i] - expected[i]) < EPSILON);
    }
    
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
    printf("✓\n");
}

void test_tensor_relu() {
    printf("Testing ReLU activation... ");
    tensor_t *input = tensor_create(1, 4);
    tensor_t *output = tensor_create(1, 4);
    
    float data[] = {-2.0f, -1.0f, 1.0f, 2.0f};
    for (int i = 0; i < 4; i++) input->data[i] = data[i];
    
    tensor_relu(input, output);
    
    float expected[] = {0.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 4; i++) {
        assert(fabsf(output->data[i] - expected[i]) < EPSILON);
    }
    
    tensor_destroy(input);
    tensor_destroy(output);
    printf("✓\n");
}

int main() {
    printf("\n Running Tensor Tests\n");
    
    test_tensor_creation();
    test_tensor_fill();
    test_tensor_add();
    test_tensor_matmul();
    test_tensor_relu();
    
    printf("\nAll tests passed!\n\n");
    return 0;
}