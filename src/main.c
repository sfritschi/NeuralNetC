#include <stdio.h>
#include <neuralnetc/neuralnet.h>

int main(void)
{
    nn_arch net;
    const uint32_t n_layers = 5;
    const uint32_t n_neurons[] = {2, 3, 4, 2, 1};

    if (nn_init(&net, n_neurons, n_layers) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize neural network\n");
        nn_free(&net);
        exit(-1);
    }
    
    nn_print(&net);
    
    nn_free(&net);

    return 0;
}
