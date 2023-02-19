#include <stdio.h>

#include <neuralnetc/neuralnet.h>
#include <neuralnetc/optim.h>
#include <neuralnetc/random_init.h>

int main(void)
{
    // Random number generator (with a given seed)
    pcg32 gen = pcg32_init();
    pcg32_seed(&gen, 21U);
    
    // Create training set (sine function + Gaussian noise)
    uint32_t i;
    const uint32_t N_train = 256;
    const uint32_t N_test  = 1024;
    const uint32_t b = 16;
    const nn_scalar_t two_pi = 2.0f * M_PI;
    const nn_scalar_t stddev = 1e-1f;
    
    // Initialize training set (allocate + preprocessing)
    nn_dataset train, test = {0};
    if (nn_dataset_init_labelled(&train, N_train, 1, b, 1) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize dataset\n");
        nn_dataset_free(&train);
        return -1;
    }
    
    if (nn_dataset_init_unlabelled(&test, N_test, 1, N_test) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize test set\n");
        nn_dataset_free(&train);
        nn_dataset_free(&test);
        return -1;
    }
    
    // x \in [0, 2*\pi], y = sin(x) + N(0, \sigma^2)
    for (i = 0; i < N_train; ++i) {
        train.data[i]   = two_pi * (nn_scalar_t)i / (nn_scalar_t)(N_train - 1);
        train.labels[i] = sinf(train.data[i]) + random_normal_scalar(&gen, 0.0, stddev);
    }
    
    nn_scalar_t *y_gt = NULL;
    CHK_ALLOC(y_gt = (nn_scalar_t *) malloc(N_test * sizeof(nn_scalar_t)));
    for (i = 0; i < N_test; ++i) {
        // Stratified random samples
        test.data[i] = two_pi * ((nn_scalar_t)i + random_uniform_scalar(&gen, 0.0, 1.0)) / (nn_scalar_t)N_test;
        y_gt[i]      = sinf(test.data[i]);
    }
    
    // Normalize training set
    if (nn_dataset_normalize(&train, NN_DATASET_NORMALIZED_MIN_MAX) != NN_E_OK) {
        fprintf(stderr, "Failed to normalize training set\n");
        nn_dataset_free(&train);
        return -1;
    }
    
    // Transfer normalization from training set to testing set
    if (nn_dataset_transfer_normalization(&train, &test) != NN_E_OK) {
        fprintf(stderr, "Failed to transfer normalization\n");
        nn_dataset_free(&train);
        nn_dataset_free(&test);
        free(y_gt);
        return -1;
    }
    
    // Write train and test sets to file
    nn_dataset_write(&train, "results/train.dat");
    
    // Initialize neural network
    nn_arch net = {0};
    const uint32_t n_layers = 4;  // 2 hidden layers
    const uint32_t n_neurons[] = {1, 16, 16, 1};
    const enum nn_activation_type activations[] = {
        NN_ACTIVATION_TANH,
        NN_ACTIVATION_TANH,
        NN_ACTIVATION_TANH
    };
    
    if (nn_init(&gen, &net, n_neurons, activations, n_layers, NN_PARAM_INIT_GLOROT) != NN_E_OK) {
        fprintf(stderr, "Failed to initialize neural network\n");
        nn_dataset_free(&train);
        nn_dataset_free(&test);
        free(y_gt);
        nn_free(&net);
        return -1;
    }
    
    // Train neural network
    const nn_scalar_t lr    = 1e-1;
    const uint32_t n_epochs = 50000;
    for (i = 0; i < n_epochs; ++i) {
        // Re-shuffle dataset
        nn_dataset_random_shuffle_samples(&gen, &train);
        // Optimization step
        nn_optim_step_SGD(&net, &train, lr, NN_LOSS_SQUARED_ERROR);
    }
    
    // Predict on test set
    if (nn_predict(&net, &test) != NN_E_OK) {
        fprintf(stderr, "Failed to predict on test set\n");
        nn_dataset_free(&train);
        nn_dataset_free(&test);
        free(y_gt);
        nn_free(&net);
        return -1;
    }
    
    // Write predicted results to file
    nn_dataset_write(&test, "results/test.dat");
    
    // Compute MSE on testing data
    const nn_loss_funcptr_t sq_error = NN_LOSS_FN[NN_LOSS_SQUARED_ERROR].l;
    
    nn_scalar_t test_error = 0.0;
    for (i = 0; i < N_test; ++i) {
        test_error += sq_error(test.labels[i], y_gt[i]);
    }
    test_error /= (nn_scalar_t)N_test;
    
    printf("MSE on test set: %.8e\n", test_error);
    
    //nn_print(&net);
    
    // Cleanup
    free(y_gt);
    
    nn_free(&net);
    nn_dataset_free(&train);
    nn_dataset_free(&test);
    
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
    pcg32 gen = pcg32_init();
    pcg32_seed(&gen, 21U);
    
    nn_arch net = {0};
    
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

    if (nn_init(&gen, &net, n_neurons, activations, n_layers, NN_PARAM_INIT_PYTORCH) != NN_E_OK) {
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
