#include <stdio.h>
#include <neuralnetc/neuralnet.h>

int main(void)
{
    nn_arch net = nn_init_empty();
    
    /* Intermediate network */
    /*
    const uint32_t n_layers = 6;
    const uint32_t n_neurons[] = {6, 3, 4, 7, 5, 2};
    const enum nn_activation_type activations[] = {
        NN_ACTIVATION_SIGMOID, 
        NN_ACTIVATION_RELU, 
        NN_ACTIVATION_SIGMOID,
        NN_ACTIVATION_TANH,
        NN_ACTIVATION_IDENTITY
    };
    */
    /* Small network */
    //const uint32_t n_layers = 3;
    //const uint32_t n_neurons[] = {1, 2, 1};
    //const nn_activation activations[] = {nn_tanh, nn_identity};
    
    /* No hidden layers */
    //const uint32_t n_layers = 2;
    //const uint32_t n_neurons[] = {1, 1};
    //const nn_activation activations[] = {nn_identity};

    /*
    if (nn_init(&net, n_neurons, activations, n_layers) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize neural network\n");
        nn_free(&net);
        return -1;
    }
    
    const nn_scalar_t x[] = {1.0, 0.0, -1.0, 0.0, 0.0, 1.0};
    
    //nn_print(&net);
    // Compute forward pass
    nn_forward(&net, x);
    
    printf("\n-- Forward Pass --\n");
    //nn_print(&net);
    
    const nn_scalar_t y[] = {0.5, 1.0};
    nn_backward(&net, y);
    
    printf("\n-- Backward Pass --\n");
    //nn_print(&net);
    
    if (nn_write(&net, "network.dat") != NN_E_OK) {
        fprintf(stderr, "Failed to write network to file\n");
        nn_free(&net);
        return -1;
    }
    */
    if (nn_read(&net, "network.dat") != NN_E_OK) {
        fprintf(stderr, "Failed to read network from file\n");
        nn_free(&net);
        return -1;
    }
    
    nn_print(&net);
    
    if (nn_write(&net, "network2.dat") != NN_E_OK) {
        fprintf(stderr, "Failed to write network to file\n");
        nn_free(&net);
        return -1;
    }
    
    nn_free(&net);

    return 0;
}
