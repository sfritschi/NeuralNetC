#include <stdio.h>
#include <neuralnetc/neuralnet.h>
#include <neuralnetc/optim.h>
#include <neuralnetc/random_init.h>

int main(void)
{
    pcg32 gen = pcg32_init();
    pcg32_seed(&gen, 21U);
    
    nn_dataset train = {0};
    if (nn_dataset_init_unlabelled(&train, 23, 2, 5) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize dataset\n");
        nn_dataset_free(&train);
        return -1;
    }
    
    nn_dataset_fill_random(&gen, &train, 1.0f, 3.0f);
    
    if (nn_dataset_normalize(&train, NN_DATASET_NORMALIZED_MIN_MAX) != NN_E_OK) {
        fprintf(stderr, "Failed to normalize training set");
        nn_dataset_free(&train);
        return -1;
    }
    
    //nn_dataset_print(&train);
    
    for (uint32_t i = 0; i < train.n_batches; ++i) {
        printf("Local batch size: %u\n", nn_dataset_local_batch_size(&train, i));
    }
    
    nn_dataset_free(&train);
    
    return 0;    
}

int main2(void)
{
    /*
    pcg32 gen = pcg32_init();
    pcg32_seed(&gen, 21U);
    
    nn_scalar_t mean, stddev;
    nn_scalar_t sum    = 0.0;
    nn_scalar_t sum_sq = 0.0;
    
    uint32_t i, N = 1000000;
    for (i = 0; i < N; ++i) {
        nn_scalar_t f = random_normal_scalar(&gen, 1.0, 2.0);
        
        sum += f;
        sum_sq += f*f;
    }
    mean = sum / (nn_scalar_t)N;
    stddev = sqrtf((sum_sq - sum*sum / (nn_scalar_t)N) / (nn_scalar_t)(N - 1));
    
    printf("Mean    = %.6f\n", mean);
    printf("Stddev. = %.6f\n", stddev);
    */
    nn_arch net = nn_init_empty();
    const uint64_t init_seed = 23U;
    
    /* Intermediate network */
    const uint32_t n_layers = 6;
    const uint32_t n_neurons[] = {6, 128, 128, 64, 32, 2};
    const enum nn_activation_type activations[] = {
        NN_ACTIVATION_TANH, 
        NN_ACTIVATION_RELU, 
        NN_ACTIVATION_SIGMOID,
        NN_ACTIVATION_TANH,
        NN_ACTIVATION_IDENTITY
    };
    /* Small network */
    //const uint32_t n_layers = 3;
    //const uint32_t n_neurons[] = {1, 2, 1};
    //const nn_activation activations[] = {nn_tanh, nn_identity};
    
    /* No hidden layers */
    //const uint32_t n_layers = 2;
    //const uint32_t n_neurons[] = {1, 1};
    //const nn_activation activations[] = {nn_identity};

    if (nn_init(&net, n_neurons, activations, n_layers, NN_PARAM_INIT_PYTORCH, init_seed) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize neural network\n");
        nn_free(&net);
        return -1;
    }
    
    const nn_scalar_t x[] = {1.0, 0.0, -1.0, 0.0, 0.0, 1.0};
    
    // Write initialized network to file
    if (nn_write(&net, "net_initial.nnc") != NN_E_OK) {
        fprintf(stderr, "Failed to write network to file\n");
        nn_free(&net);
        return -1;
    }
    
    // Compute forward pass
    nn_forward(&net, x);
    
    //printf("\n-- Forward Pass --\n");
    //nn_print(&net);
    
    const nn_scalar_t y[] = {0.5, 1.0};
    nn_backward(&net, y, NN_LOSS_SQUARED_ERROR);
    
    //printf("\n-- Backward Pass --\n");
    //nn_print(&net);
    
    // Write network after backward pass to file
    if (nn_write(&net, "net_backward.nnc") != NN_E_OK) {
        fprintf(stderr, "Failed to write network to file\n");
        nn_free(&net);
        return -1;
    }
        
    /*
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
    */
    
    nn_free(&net);
    return 0;
}
