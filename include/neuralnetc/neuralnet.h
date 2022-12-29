#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdint.h>
#include <stdlib.h>
#include <neuralnetc/activation.h>

enum nn_errors {
    NN_E_OK = 0,
    NN_E_TOO_FEW_LAYERS,
    NN_E_OUT_OF_MEM
};

#define CHK_ALLOC(ptr) {\
    if (!(ptr)) {\
        return NN_E_OUT_OF_MEM;\
    }\
}

typedef struct {
    uint32_t n_hidden_layers;
    uint32_t *n_neurons;
    nn_scalar_t *weights;
    uint32_t *offsets_weights;
    nn_scalar_t *biases;
    uint32_t *offsets_biases;
    nn_funcptr_t *activations;    
} nn_arch;

void nn_init_params(nn_arch *net)
{
    // Iterate through weights and biases and initialize them to 0 for now
    uint32_t i, l, ow_prev, ow_next, ob_prev, ob_next;
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
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
    }
}

int nn_init(nn_arch *net, const uint32_t *n_neurons, uint32_t n_layers)
{
    // Set all pointers to NULL
    net->n_neurons = NULL;
    net->weights = NULL;
    net->offsets_weights = NULL;
    net->biases = NULL;
    net->offsets_biases = NULL;
    net->activations = NULL;
    
    if (n_layers < 2) {
        return NN_E_TOO_FEW_LAYERS;
    }
    
    const uint32_t n_layers_size = n_layers * sizeof(uint32_t);

    net->n_hidden_layers = n_layers - 2;
    // Allocate arrays
    CHK_ALLOC(net->n_neurons = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->offsets_weights = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->offsets_biases  = (uint32_t *) malloc(n_layers_size));
    // For now, initialize to 0 (identity activation)
    CHK_ALLOC(net->activations = (nn_funcptr_t *) calloc(n_layers-1, sizeof(nn_funcptr_t)));
    // Copy number of neurons for each layer
    uint32_t total_weights = 0;
    uint32_t total_biases  = 0;

    // Initialize offsets to 0
    net->offsets_weights[0] = 0;
    net->offsets_biases[0]  = 0;

    uint32_t i;
    for (i = 0; i < n_layers-1; ++i) {
        const uint32_t neurons_prev = n_neurons[i];
        const uint32_t neurons_next = n_neurons[i+1];
 
        net->n_neurons[i] = neurons_prev;
        total_weights += neurons_prev * neurons_next;
        total_biases  += neurons_next;

        // Set offsets in weights and biases arrays
        net->offsets_weights[i+1] = total_weights;
        net->offsets_biases[i+1]  = total_biases;
    }
    // Number of neurons in output
    net->n_neurons[i] = n_neurons[i];

    // Allocate weights and biases
    CHK_ALLOC(net->weights = (nn_scalar_t *) malloc(total_weights * sizeof(nn_scalar_t)));
    CHK_ALLOC(net->biases  = (nn_scalar_t *) malloc(total_biases * sizeof(nn_scalar_t)));

    nn_init_params(net);
    
    return NN_E_OK;
}

void nn_print(const nn_arch *net)
{
    printf("#Hidden Layers: %u\n\n", net->n_hidden_layers);
    uint32_t i, j, l, ow, ob_prev, ob_next, rows, cols;
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        ow = net->offsets_weights[l];
        ob_prev = net->offsets_biases[l];
        ob_next = net->offsets_biases[l+1];

        printf("Layer #%u\n", l+1);
        rows = net->n_neurons[l+1];
        cols = net->n_neurons[l];
        printf("Weights: (%u x %u)\n", rows, cols);
        for (i = 0; i < rows; ++i) {
            for (j = 0; j < cols; ++j) {
                printf("%.3f ", net->weights[i*cols + j + ow]);
            }
            printf("\n");
        }
        
        printf("Biases: %u\n", ob_next - ob_prev);
        for (i = ob_prev; i < ob_next; ++i) {
            printf("%.3f ", net->biases[i]);
        }
        printf("\n\n");
    }
}

void nn_free(nn_arch *net)
{
    free(net->n_neurons);
    free(net->weights);
    free(net->biases);
    free(net->offsets_weights);
    free(net->offsets_biases);
    free(net->activations);
}

#endif /* NEURAL_NET_H */
