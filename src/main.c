#include <stdio.h>
#include <neuralnetc/neuralnet.h>

int main(void)
{
    nn_arch net;
    
    const uint32_t n_layers = 5;
    const uint32_t n_neurons[] = {2, 3, 4, 2, 1};
    const nn_activation activations[] = {nn_ReLU, nn_ReLU, nn_ReLU, nn_identity};
    
    if (nn_init(&net, n_neurons, activations, n_layers) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize neural network\n");
        nn_free(&net);
        return -1;
    }

    nn_print(&net);
    // Compute forward pass
    nn_forward(&net);
    nn_print(&net);
    
    nn_free(&net);

    return 0;
}
