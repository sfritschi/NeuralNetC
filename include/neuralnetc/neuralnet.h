#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdint.h>
#include <stdlib.h>

#include <assert.h>

#include <neuralnetc/common.h>
#include <neuralnetc/activation.h>

// Differentiable variable type
typedef struct {
    nn_scalar_t value;
    nn_scalar_t grad;    
} nn_diffable_t;

typedef struct {
    uint32_t n_hidden_layers;  // n_layers - 2
    nn_diffable_t *neurons;
    uint32_t *n_neurons;
    uint32_t *offsets_neurons;
    nn_diffable_t *weights;
    uint32_t *offsets_weights;
    nn_diffable_t *biases;
    uint32_t *offsets_biases;
    nn_activation *activations;
    nn_scalar_t *error_signals;  // buffer needed for back propagation
} nn_arch;

void nn_init_params(nn_arch *net)
{
    // TODO: Initialize weights and biases randomly based on activation function
    //       using PCG32 random number generator
    uint32_t i, l, on_prev, on_next, ow_prev, ow_next, ob_prev, ob_next;
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        on_prev = net->offsets_neurons[l];
        on_next = net->offsets_neurons[l+1];
        ow_prev = net->offsets_weights[l];
        ow_next = net->offsets_weights[l+1];
        ob_prev = net->offsets_biases[l];
        ob_next = net->offsets_biases[l+1];

        for (i = ow_prev; i < ow_next; ++i) {
            net->weights[i] = (nn_diffable_t) {i, 0.0};
        }

        for (i = ob_prev; i < ob_next; ++i) {
            net->biases[i] = (nn_diffable_t) {0.0, 0.0};
            net->error_signals[i] = (nn_scalar_t) 0.0;
        }
        
        for (i = on_prev; i < on_next; ++i) {
            net->neurons[i] = (nn_diffable_t) {0.0, 0.0};
        }
    }
    
    // Set neurons of output layer
    on_prev = net->offsets_neurons[l];
    on_next = net->offsets_neurons[l+1];
    for (i = on_prev; i < on_next; ++i) {
        net->neurons[i] = (nn_diffable_t) {0.0, 0.0};
    }
}

int nn_init(nn_arch *net, const uint32_t *n_neurons, 
            const nn_activation *activations, uint32_t n_layers)
{
    // Set all pointers to NULL
    net->neurons = NULL;
    net->offsets_neurons = NULL;
    net->n_neurons = NULL;
    net->weights = NULL;
    net->offsets_weights = NULL;
    net->biases = NULL;
    net->error_signals = NULL;
    net->offsets_biases = NULL;
    net->activations = NULL;
    
    if (n_layers < 2) {
        return NN_E_TOO_FEW_LAYERS;
    }
    // Size of layers in bytes
    const uint32_t n_layers_size = n_layers * sizeof(uint32_t);

    net->n_hidden_layers = n_layers - 2;
    // Allocate arrays
    CHK_ALLOC(net->n_neurons = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->offsets_neurons = (uint32_t *) malloc((n_layers+1)*sizeof(uint32_t)));
    CHK_ALLOC(net->offsets_weights = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->offsets_biases  = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->activations = (nn_activation *) malloc((n_layers-1)*sizeof(nn_activation)));
    // Copy number of neurons for each layer
    uint32_t total_neurons = 0;
    uint32_t total_weights = 0;
    uint32_t total_biases  = 0;

    // Initialize offsets to 0
    net->offsets_neurons[0] = 0;
    net->offsets_weights[0] = 0;
    net->offsets_biases[0]  = 0;

    uint32_t l;
    for (l = 0; l < n_layers-1; ++l) {
        const uint32_t neurons_prev = n_neurons[l];
        const uint32_t neurons_next = n_neurons[l+1];
 
        // Initialize number of neurons and activation functions per layer
        net->activations[l] = activations[l];
        net->n_neurons[l] = neurons_prev;
        total_neurons += neurons_prev;
        total_weights += neurons_prev * neurons_next;
        total_biases  += neurons_next;

        // Set offsets in weights and biases arrays
        net->offsets_neurons[l+1] = total_neurons;
        net->offsets_weights[l+1] = total_weights;
        net->offsets_biases[l+1]  = total_biases;
    }
    // Number of neurons in output
    total_neurons += n_neurons[l];
    net->n_neurons[l] = n_neurons[l];
    net->offsets_neurons[l+1] = total_neurons;
    
    // Allocate remaining buffers
    CHK_ALLOC(net->neurons = (nn_diffable_t *) malloc(total_neurons * sizeof(nn_diffable_t)));
    CHK_ALLOC(net->weights = (nn_diffable_t *) malloc(total_weights * sizeof(nn_diffable_t)));
    CHK_ALLOC(net->biases  = (nn_diffable_t *) malloc(total_biases * sizeof(nn_diffable_t)));
    // Note: Error signals have same exact dimensions as biases
    CHK_ALLOC(net->error_signals = (nn_scalar_t *) malloc(total_biases * sizeof(nn_scalar_t)));

    nn_init_params(net);
    
    return NN_E_OK;
}

// Compute forward pass to evaluate neural network architecture
void nn_forward(nn_arch *net, const nn_scalar_t *x)
{
    nn_scalar_t sum, activation, grad;

    uint32_t i, j, l, on, on_next, ow, ob, rows, cols;
    
    // Set function argument at input layer
    for (i = 0; i < net->n_neurons[0]; ++i) {
        // offset is 0, since first layer
        net->neurons[i].value = x[i];
    }
    
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        on_next = net->offsets_neurons[l+1];
        on      = net->offsets_neurons[l];
        ow      = net->offsets_weights[l];
        ob      = net->offsets_biases[l];
        rows    = net->n_neurons[l+1];
        cols    = net->n_neurons[l];
        
        for (i = 0; i < rows; ++i) {
            sum = (nn_scalar_t) 0.0;
            for (j = 0; j < cols; ++j) {
                // Matrix-vector product between weights and neurons
                sum += net->weights[i*cols + j + ow].value * net->neurons[j + on].value;
            }
            // Add the bias for this neuron
            sum += net->biases[i + ob].value;
            
            activation = sum;
            grad = (nn_scalar_t) 1.0;  // identity activation gradient
            // Apply activation function, if any
            const nn_activation act = net->activations[l];
            if (act.f && act.gradf) {
                activation = act.f(activation);
                grad       = act.gradf(activation);
            }
            
            net->neurons[i + on_next] = (nn_diffable_t) {
                activation,
                grad
            };
        }
    }
}

void nn_backward(nn_arch *net, const nn_scalar_t *y_label)
{
    uint32_t i, j, k, l, on_next, on, ow_next, ow, ob_next, ob, rows, rows_next, cols;
    // Compute error signal at final layer
    on_next = net->offsets_neurons[net->n_hidden_layers+1];
    on      = net->offsets_neurons[net->n_hidden_layers];
    ob      = net->offsets_biases[net->n_hidden_layers];  // error signal offset
    ow      = net->offsets_weights[net->n_hidden_layers];
    rows    = net->n_neurons[net->n_hidden_layers+1];
    cols    = net->n_neurons[net->n_hidden_layers];
    
    nn_scalar_t error_signal;
    
    for (i = 0; i < rows; ++i) {
        // Note: Implicitly assumes squared error loss
        // TODO: Add loss function structure to nn_arch
        const nn_diffable_t neuron = net->neurons[i + on_next];
        // del_L,i
        error_signal = (nn_scalar_t)2.0 * (neuron.value - y_label[i]) * neuron.grad;
        net->error_signals[i + ob] = error_signal;
        
        for (j = 0; j < cols; ++j) {
            // grad(w)_ij,L = del_L,i * f_L-1,j
            net->weights[i*cols + j + ow].grad = error_signal * 
                net->neurons[j + on].value;
        }
        // grad(b)_i,L = del_L,i
        net->biases[i + ob].grad = error_signal;
    }
    
    // Iterate over previous layers to propagate error signal backwards
    for (l = net->n_hidden_layers-1 ;; --l) {
        on_next   = net->offsets_neurons[l+1];
        on        = net->offsets_neurons[l];
        ob_next   = net->offsets_biases[l+1];
        ob        = net->offsets_biases[l];  // error signal offset
        ow_next   = net->offsets_weights[l+1];
        ow        = net->offsets_weights[l];
        rows_next = net->n_neurons[l+2];
        rows      = net->n_neurons[l+1];
        cols      = net->n_neurons[l];
        
        for (i = 0; i < rows; ++i) {
            error_signal = (nn_scalar_t) 0.0;
            
            // Compute current error signal using last visited layer
            for (k = 0; k < rows_next; ++k) {
                // del_l,i += del_l+1,k * w_ki,l+1
                error_signal += net->error_signals[k + ob_next] * 
                    net->weights[k*rows + i + ow_next].value;
            }
            // del_l,i = del_l,i * grad(f)_l,i
            error_signal *= net->neurons[i + on_next].grad;
            net->error_signals[i + ob] = error_signal;
            
            for (j = 0; j < cols; ++j) {
                // grad(w)_ij,l = del_l,i * f_l-1,j
                net->weights[i*cols + j + ow].grad = error_signal * 
                    net->neurons[j + on].value;
            }
            // grad(b)_i,l = del_l,i
            net->biases[i + ob].grad = error_signal;
        }
        
        // Reached first layer; done
        if (l == 0) break;
    }
}

void nn_print(const nn_arch *net)
{
    printf("#Hidden Layers: %u\n\n", net->n_hidden_layers);
    uint32_t i, j, l, on_prev, on_next, ow, ob_prev, ob_next, rows, cols;
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        on_prev = net->offsets_neurons[l];
        on_next = net->offsets_neurons[l+1];
        ow = net->offsets_weights[l];
        ob_prev = net->offsets_biases[l];
        ob_next = net->offsets_biases[l+1];

        printf("Layer #%u\n", l+1);
        printf("Activation: %s\n", NN_ACTIVATION_NAMES[net->activations[l].type]);
        rows = net->n_neurons[l+1];
        cols = net->n_neurons[l];
        printf("#Weights: (%u x %u)\n", rows, cols);
        for (i = 0; i < rows; ++i) {
            for (j = 0; j < cols; ++j) {
                const uint32_t index = i*cols + j + ow;
                printf("(v: %.3f, g: %.3f) ", net->weights[index].value,
                                              net->weights[index].grad);
            }
            printf("\n");
        }
        
        printf("#Biases: %u\n", ob_next - ob_prev);
        for (i = ob_prev; i < ob_next; ++i) {
            printf("(v: %.3f, g: %.3f) ", net->biases[i].value, 
                                         net->biases[i].grad);
        }
        printf("\n");
        
        printf("#Neurons: %u\n", on_next - on_prev);
        for (i = on_prev; i < on_next; ++i) {
            printf("(v: %.3f, g: %.3f) ", net->neurons[i].value, 
                                          net->neurons[i].grad);
        }
        printf("\n\n");
    }
    
    printf("Layer #%u\n", l+1);
    on_prev = net->offsets_neurons[l];
    on_next = net->offsets_neurons[l+1];
    printf("#Neurons: %u\n", on_next - on_prev);
    for (i = on_prev; i < on_next; ++i) {
        printf("(v: %.3f, g: %.3f) ", net->neurons[i].value, 
                                      net->neurons[i].grad);
    }
    printf("\n\n");
}

void nn_free(nn_arch *net)
{
    net->n_hidden_layers = 0;
    
    NN_FREE_NULL(net->neurons);
    NN_FREE_NULL(net->offsets_neurons);
    NN_FREE_NULL(net->n_neurons);
    NN_FREE_NULL(net->weights);
    NN_FREE_NULL(net->biases);
    NN_FREE_NULL(net->error_signals);
    NN_FREE_NULL(net->offsets_weights);
    NN_FREE_NULL(net->offsets_biases);
    NN_FREE_NULL(net->activations);
}

#endif /* NEURAL_NET_H */
