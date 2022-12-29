#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdint.h>
#include <stdlib.h>

#include <neuralnetc/common.h>
#include <neuralnetc/activation.h>

typedef struct {
    uint32_t n_hidden_layers;  // n_layers - 2
    nn_scalar_t *neurons;
    uint32_t *n_neurons;
    uint32_t *offsets_neurons;
    nn_scalar_t *weights;
    uint32_t *offsets_weights;
    nn_scalar_t *biases;
    uint32_t *offsets_biases;
    nn_activation *activations;    
} nn_arch;

void nn_init_params(nn_arch *net)
{
    // Iterate through weights and biases and initialize them to 0 for now
    uint32_t i, l, on_prev, on_next, ow_prev, ow_next, ob_prev, ob_next;
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        on_prev = net->offsets_neurons[l];
        on_next = net->offsets_neurons[l+1];
        ow_prev = net->offsets_weights[l];
        ow_next = net->offsets_weights[l+1];
        ob_prev = net->offsets_biases[l];
        ob_next = net->offsets_biases[l+1];

        for (i = ow_prev; i < ow_next; ++i) {
            net->weights[i] = (nn_scalar_t) i;
        }

        for (i = ob_prev; i < ob_next; ++i) {
            net->biases[i] = (nn_scalar_t) i;
        }
        
        for (i = on_prev; i < on_next; ++i) {
            net->neurons[i] = (nn_scalar_t) 0.0;
        }
    }
    
    // Set neurons of output layer
    on_prev = net->offsets_neurons[l];
    on_next = net->offsets_neurons[l+1];
    for (i = on_prev; i < on_next; ++i) {
        net->neurons[i] = (nn_scalar_t) 0.0;
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
    
    // Allocate neurons, weights and biases
    CHK_ALLOC(net->neurons = (nn_scalar_t *) malloc(total_neurons * sizeof(nn_scalar_t)));
    CHK_ALLOC(net->weights = (nn_scalar_t *) malloc(total_weights * sizeof(nn_scalar_t)));
    CHK_ALLOC(net->biases  = (nn_scalar_t *) malloc(total_biases * sizeof(nn_scalar_t)));

    nn_init_params(net);
    
    return NN_E_OK;
}

// Compute forward pass to evaluate neural network architecture
void nn_forward(nn_arch *net)
{
    nn_scalar_t sum;
    
    uint32_t i, j, l, on, on_next, ow, ob, rows, cols;
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        on      = net->offsets_neurons[l];
        on_next = net->offsets_neurons[l+1];
        ow      = net->offsets_weights[l];
        ob      = net->offsets_biases[l];
        
        rows = net->n_neurons[l+1];
        cols = net->n_neurons[l];
        for (i = 0; i < rows; ++i) {
            sum = (nn_scalar_t) 0.0;
            for (j = 0; j < cols; ++j) {
                // Matrix product between weights and neurons
                sum += net->weights[i*cols + j + ow] * net->neurons[j + on];
            }
            // Add the bias for this neuron
            sum += net->biases[i + ob];
            // Apply activation function, if any
            if (net->activations[l].f)
                sum = net->activations[l].f(sum);
            
            net->neurons[i + on_next] = sum;
        }
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
                printf("%.3f ", net->weights[i*cols + j + ow]);
            }
            printf("\n");
        }
        
        printf("#Biases: %u\n", ob_next - ob_prev);
        for (i = ob_prev; i < ob_next; ++i) {
            printf("%.3f ", net->biases[i]);
        }
        printf("\n");
        
        printf("#Neurons: %u\n", on_next - on_prev);
        for (i = on_prev; i < on_next; ++i) {
            printf("%.3f ", net->neurons[i]);
        }
        printf("\n\n");
    }
    
    printf("Layer #%u\n", l+1);
    on_prev = net->offsets_neurons[l];
    on_next = net->offsets_neurons[l+1];
    printf("#Neurons: %u\n", on_next - on_prev);
    for (i = on_prev; i < on_next; ++i) {
        printf("%.3f ", net->neurons[i]);
    }
    printf("\n\n");
}

void nn_free(nn_arch *net)
{
    free(net->neurons);
    free(net->offsets_neurons);
    free(net->n_neurons);
    free(net->weights);
    free(net->biases);
    free(net->offsets_weights);
    free(net->offsets_biases);
    free(net->activations);
}

#endif /* NEURAL_NET_H */
