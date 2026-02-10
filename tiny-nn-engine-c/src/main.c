#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"

int main() {
    printf("🧠 Tiny Neural Network Engine - XOR Problem\n");
    float xor_inputs_data[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    float xor_targets_data[4][1] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };
    
    tensor_t *X = tensor_create(4, 2);
    tensor_t *y = tensor_create(4, 1);
    
    for (int i = 0; i < 4; i++) {
        X->data[i * 2 + 0] = xor_inputs_data[i][0];
        X->data[i * 2 + 1] = xor_inputs_data[i][1];
        y->data[i] = xor_targets_data[i][0];
    }
    
    dense_layer_t *layer1 = layer_create(2, 4, ACTIVATION_RELU);
    dense_layer_t *layer2 = layer_create(4, 1, ACTIVATION_SIGMOID);
    dense_layer_t *layers[] = {layer1, layer2};
    adam_optimizer_t *optimizer = adam_create(0.1f, 2);
    int epochs = 5000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        tensor_t *hidden = layer_forward(layer1, X);
        tensor_t *output = layer_forward(layer2, hidden);
        float loss = loss_binary_crossentropy(output, y);
        tensor_t *grad = tensor_create(output->rows, output->cols);
        loss_bce_derivative(output, y, grad);   
        tensor_t *grad_hidden = layer_backward(layer2, grad);
        tensor_t *grad_input = layer_backward(layer1, grad_hidden);
        adam_step(optimizer, layers, 2);
        tensor_destroy(grad);
        tensor_destroy(grad_hidden);
        tensor_destroy(grad_input);
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %5d | Loss: %.6f\n", epoch + 1, loss);
        }
    }
    
    printf("\n✅ Training Complete!\n\n");
    printf("XOR Predictions:\n");
    printf("----------------\n");
    tensor_t *final_hidden = layer_forward(layer1, X);
    tensor_t *predictions = layer_forward(layer2, final_hidden);
    
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float pred = predictions->data[i];
        float target = y->data[i];
        int predicted_class = pred > 0.5f ? 1 : 0;
        int target_class = (int)target;
        
        printf("Input: [%.0f, %.0f] | Predicted: %.4f | Target: %.0f | %s\n",
               X->data[i * 2 + 0], X->data[i * 2 + 1],
               pred, target,
               predicted_class == target_class ? "✓" : "✗");
        
        if (predicted_class == target_class) correct++;
    }
    
    printf("\nAccuracy: %d/4 (%.1f%%)\n", correct, (correct / 4.0f) * 100.0f);
    tensor_destroy(X);
    tensor_destroy(y);
    layer_destroy(layer1);
    layer_destroy(layer2);
    adam_destroy(optimizer);
    
    printf("\n🚀 Program complete!\n");
    
    return 0;
}