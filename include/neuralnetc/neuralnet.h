#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include <neuralnetc/common.h>
#include <neuralnetc/activation.h>
#include <neuralnetc/random_init.h>

#define CHK_WRITE(buf, size, nmemb, fp) do {\
    if (fwrite((buf), (size), (nmemb), (fp)) != (nmemb)) {\
        fclose((fp));\
        return NN_E_FAILED_TO_WRITE_FILE;\
    }\
} while(0)

#if __BYTE_ORDER == __LITTLE_ENDIAN
#define CHK_READ(buf, size, nmemb, fp) do {\
    if (fread((buf), (size), (nmemb), (fp)) != (nmemb)) {\
        fclose((fp));\
        return NN_E_FAILED_TO_READ_FILE;\
    }\
} while(0)
#elif __BYTE_ORDER == __BIG_ENDIAN
#define CHK_READ(buf, size, nmemb, fp) do {\
    if (fread((buf), (size), (nmemb), (fp)) != (nmemb)) {\
        fclose((fp));\
        return NN_E_FAILED_TO_READ_FILE;\
    }\
    {\
        uint32_t i, j;\
        for (i = 0; i < (nmemb); ++i) {\
            uint8_t *bytes = (uint8_t *)&(buf)[i];\
            for (j = 0; j < (size) / 2; ++j) {\
                const uint8_t temp = bytes[j];\
                bytes[j] = bytes[(size)-1-j];\
                bytes[(size)-1-j] = temp;\
            }\
        }\
    }\
} while(0)
#else
#error "Unrecognized byte order"
#endif

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
    nn_scalar_t *error_signals;  // buffer needed for backpropagation
    uint8_t is_initialized;
} nn_arch;

static void nn_init_params(nn_arch *net, uint64_t seed)
{
    uint32_t i, l, on_prev, on_next, ow_prev, ow_next, ob_prev, ob_next;
    uint32_t n_in, n_out;
    nn_scalar_t stddev;
    
    // ?TODO: Determine if random number generator should be global
    pcg32 gen = pcg32_init();
    pcg32_seed(&gen, seed);
    
    for (l = 0; l < net->n_hidden_layers + 1; ++l) {
        on_prev = net->offsets_neurons[l];
        on_next = net->offsets_neurons[l+1];
        ow_prev = net->offsets_weights[l];
        ow_next = net->offsets_weights[l+1];
        ob_prev = net->offsets_biases[l];
        ob_next = net->offsets_biases[l+1];

        n_in  = net->n_neurons[l];
        n_out = net->n_neurons[l+1];
        
        // Glorot (random) weight-initialization standard deviation (tanh)
        stddev = sqrtf((nn_scalar_t)2.0 / (nn_scalar_t)(n_in + n_out));
        
        for (i = ow_prev; i < ow_next; ++i) {
            net->weights[i] = (nn_diffable_t) {
                random_normal_scalar(&gen, 0.0, stddev),
                0.0
            };
        }
        // Initialize all biases to 0
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

// Return empty network
nn_arch nn_init_empty() {
    return (nn_arch) {0};
}

int nn_init(nn_arch *net, const uint32_t *n_neurons, 
            const enum nn_activation_type *activations, uint32_t n_layers,
            uint64_t seed)
{
    if (!net || net->is_initialized)
        return NN_E_NET_ALREADY_INITIALIZED;
    
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

    // Initialize start offsets to 0
    net->offsets_neurons[0] = 0;
    net->offsets_weights[0] = 0;
    net->offsets_biases[0]  = 0;

    uint32_t l;
    for (l = 0; l < n_layers-1; ++l) {
        const uint32_t neurons_prev = n_neurons[l];
        const uint32_t neurons_next = n_neurons[l+1];
 
        if (neurons_prev < 1 || neurons_next < 1) {
            return NN_E_TOO_FEW_NEURONS;
        }
        // Initialize number of neurons and activation functions per layer
        net->activations[l] = NN_ACTIVATIONS[activations[l]];
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

    nn_init_params(net, seed);
    
    // Set initialization flag
    net->is_initialized = 1;
    
    return NN_E_OK;
}

// Compute forward pass to evaluate neural network architecture
int nn_forward(nn_arch *net, const nn_scalar_t *x)
{
    if (!net || !net->is_initialized)
        return NN_E_NET_UNINITIALIZED;
    
    nn_scalar_t sum, activation, grad;

    uint32_t i, j, l, on, on_next, ow, ob, rows, cols;
    
    // Set function argument at input layer
    for (i = 0; i < net->n_neurons[0]; ++i) {
        // offset is 0, since we are in the first layer
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
    
    return NN_E_OK;
}

int nn_backward(nn_arch *net, const nn_scalar_t *y_label)
{
    if (!net || !net->is_initialized)
        return NN_E_NET_UNINITIALIZED;
        
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
            // grad(w)_ij,L = del_i,L * f_j,L-1
            net->weights[i*cols + j + ow].grad = error_signal * 
                net->neurons[j + on].value;
        }
        // grad(b)_i,L = del_i,L
        net->biases[i + ob].grad = error_signal;
    }

    if (net->n_hidden_layers == 0)
        return NN_E_OK;  // no hidden layers; done
        
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
                // del_i,l += del_k,l+1 * w_ki,l+1
                error_signal += net->error_signals[k + ob_next] * 
                    net->weights[k*rows + i + ow_next].value;
            }
            // del_l,i = del_i,l * grad(f)_i,l
            error_signal *= net->neurons[i + on_next].grad;
            // NOTE: No need to store error signals for ALL layers.
            //       Only required for previous layer
            net->error_signals[i + ob] = error_signal;
            
            for (j = 0; j < cols; ++j) {
                // grad(w)_ij,l = del_i,l * f_j,l-1
                net->weights[i*cols + j + ow].grad = error_signal * 
                    net->neurons[j + on].value;
            }
            // grad(b)_i,l = del_i,l
            net->biases[i + ob].grad = error_signal;
        }
        
        // Reached first layer; done
        if (l == 0) break;
    }
    
    return NN_E_OK;
}

int nn_print(const nn_arch *net)
{
    if (!net || !net->is_initialized)
        return NN_E_NET_UNINITIALIZED;
        
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
                printf("(v: %.4e, g: %.4e) ", net->weights[index].value,
                                              net->weights[index].grad);
            }
            printf("\n");
        }
        
        printf("#Biases: %u\n", ob_next - ob_prev);
        for (i = ob_prev; i < ob_next; ++i) {
            printf("(v: %.4e, g: %.4e) ", net->biases[i].value, 
                                          net->biases[i].grad);
        }
        printf("\n");
        
        printf("#Neurons: %u\n", on_next - on_prev);
        for (i = on_prev; i < on_next; ++i) {
            printf("(v: %.4e, g: %.4e) ", net->neurons[i].value, 
                                          net->neurons[i].grad);
        }
        printf("\n\n");
    }
    
    printf("Layer #%u\n", l+1);
    on_prev = net->offsets_neurons[l];
    on_next = net->offsets_neurons[l+1];
    printf("#Neurons: %u\n", on_next - on_prev);
    for (i = on_prev; i < on_next; ++i) {
        printf("(v: %.4e, g: %.4e) ", net->neurons[i].value, 
                                      net->neurons[i].grad);
    }
    printf("\n\n");
    
    return NN_E_OK;
}

/*
 * Writes all relevant data to binary file needed to reconstruct network.
 * 
 * File format (binary, little-endian):
 * - 'NNC' signature
 * - #layers (uint32_t)
 * - #neurons for each layer (uint32_t)
 * - enum codes (integers) associated with activation functions
 * - neurons (nn_diffable_t)
 * - weights (nn_diffable_t)
 * - biases (nn_diffable_t)
 * - error signals (nn_scalar_t)
 *
 */
int nn_write(const nn_arch *net, const char *filename)
{
    if (!net || !net->is_initialized)
        return NN_E_NET_UNINITIALIZED;
        
    FILE *fp = fopen(filename, "wb");
    if (!fp) return NN_E_FAILED_TO_WRITE_FILE;
    
    uint32_t i;
    const uint32_t n_layers = net->n_hidden_layers + 2;
    const uint32_t total_neurons = net->offsets_neurons[n_layers];
    const uint32_t total_weights = net->offsets_weights[n_layers-1];
    const uint32_t total_biases  = net->offsets_biases[n_layers-1];
    
    CHK_WRITE(FILE_SIGNATURE, 1, SIGNATURE_LEN, fp);
    
    // Check if architecture is little-endian
    #if __BYTE_ORDER == __LITTLE_ENDIAN
    CHK_WRITE(&n_layers, sizeof(uint32_t), 1, fp);
    CHK_WRITE(net->n_neurons, sizeof(uint32_t), n_layers, fp);
    
    for (i = 0; i < n_layers-1; ++i) {
        CHK_WRITE(&net->activations[i].type, 
            sizeof(net->activations[0].type), 1, fp);
    }
    
    CHK_WRITE(net->neurons, sizeof(net->neurons[0]), total_neurons, fp);
    CHK_WRITE(net->weights, sizeof(net->weights[0]), total_weights, fp);
    CHK_WRITE(net->biases, sizeof(net->biases[0]), total_biases, fp);
    CHK_WRITE(net->error_signals, sizeof(net->error_signals[0]), total_biases, fp);
    
    #elif __BYTE_ORDER == __BIG_ENDIAN
    #define REVERSE_WRITE(data, nelems, fp) do {\
        const uint32_t size = sizeof((data)[0]);\
        for (uint32_t i = 0; i < (nelems); ++i) {\
            const uint8_t *bytes = (const uint8_t *)&(data)[i];\
            for (uint32_t j = 0; j < size; ++j) {\
                CHK_WRITE(&bytes[size - 1 - j], sizeof(uint8_t), 1, (fp));\
            }\
        }\
    } while(0)
    
    // Big endian byte order -> swap bytes
    REVERSE_WRITE(&n_layers, 1, fp);
    // Write reversed number of neurons to file
    REVERSE_WRITE(net->n_neurons, n_layers, fp);
    
    // TODO: Resolve issue with index variable 'i' in macro shadowing local variable
    for (uint32_t index = 0; index < n_layers-1; ++index) {
        REVERSE_WRITE(&net->activations[index].type, 1, fp);
    }
    
    // Write neurons, weights, biases and error signals to file
    REVERSE_WRITE(net->neurons, total_neurons, fp);
    REVERSE_WRITE(net->weights, total_weights, fp);
    REVERSE_WRITE(net->biases, total_biases, fp);
    REVERSE_WRITE(net->error_signals, total_biases, fp);
    
    #else
    #error "Unrecognized byte order"
    #endif
    
    fclose(fp);
    return NN_E_OK;
}

/*
 *  Reads network from given file
 */
int nn_read(nn_arch *net, const char *filename)
{
    if (!net || net->is_initialized)
        return NN_E_NET_ALREADY_INITIALIZED;
        
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NN_E_FAILED_TO_READ_FILE;
    
    // Read signature
    uint32_t i;
    char signature_buf[SIGNATURE_LEN];
    CHK_READ(signature_buf, 1, SIGNATURE_LEN, fp);
    
    for (i = 0; i < SIGNATURE_LEN; ++i) {
        if (signature_buf[i] != FILE_SIGNATURE[i]) {
            fclose(fp);
            return NN_E_UNRECOGNIZED_READ_SIGNATURE;
        }
    }
        
    // Read number of layers
    uint32_t n_layers;
    CHK_READ(&n_layers, sizeof(uint32_t), 1, fp);
    if (n_layers < 2) {
        fclose(fp);
        return NN_E_TOO_FEW_LAYERS;
    }
    const uint32_t n_layers_size = n_layers * sizeof(uint32_t);
    
    net->n_hidden_layers = n_layers - 2;
    // Allocate arrays
    CHK_ALLOC(net->n_neurons = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->offsets_neurons = (uint32_t *) malloc((n_layers+1)*sizeof(uint32_t)));
    CHK_ALLOC(net->offsets_weights = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->offsets_biases  = (uint32_t *) malloc(n_layers_size));
    CHK_ALLOC(net->activations = (nn_activation *) malloc((n_layers-1)*sizeof(nn_activation)));
    
    // Read number of neurons per layer
    CHK_READ(net->n_neurons, sizeof(net->n_neurons[0]), n_layers, fp);
    // Compute offsets for each layer
    uint32_t total_neurons = 0;
    uint32_t total_weights = 0;
    uint32_t total_biases  = 0;

    // Initialize offsets to 0
    net->offsets_neurons[0] = 0;
    net->offsets_weights[0] = 0;
    net->offsets_biases[0]  = 0;
    
    // Read enum codes of activation functions and compute offsets
    for (i = 0; i < n_layers-1; ++i) {
        enum nn_activation_type activation;
        CHK_READ(&activation, sizeof(enum nn_activation_type), 1, fp);
        net->activations[i] = NN_ACTIVATIONS[activation];
        
        const uint32_t neurons_prev = net->n_neurons[i];
        const uint32_t neurons_next = net->n_neurons[i+1];
 
        total_neurons += neurons_prev;
        total_weights += neurons_prev * neurons_next;
        total_biases  += neurons_next;

        // Set offsets in weights and biases arrays
        net->offsets_neurons[i+1] = total_neurons;
        net->offsets_weights[i+1] = total_weights;
        net->offsets_biases[i+1]  = total_biases;
    }
    // Number of neurons in output
    total_neurons += net->n_neurons[i];
    net->offsets_neurons[i+1] = total_neurons;
    
    // Allocate & read remaining buffers
    CHK_ALLOC(net->neurons = (nn_diffable_t *) malloc(total_neurons * sizeof(nn_diffable_t)));
    CHK_ALLOC(net->weights = (nn_diffable_t *) malloc(total_weights * sizeof(nn_diffable_t)));
    CHK_ALLOC(net->biases  = (nn_diffable_t *) malloc(total_biases * sizeof(nn_diffable_t)));
    // Note: Error signals have same exact dimensions as biases
    CHK_ALLOC(net->error_signals = (nn_scalar_t *) malloc(total_biases * sizeof(nn_scalar_t)));
    
    CHK_READ(net->neurons, sizeof(net->neurons[0]), total_neurons, fp);
    CHK_READ(net->weights, sizeof(net->weights[0]), total_weights, fp);
    CHK_READ(net->biases, sizeof(net->biases[0]), total_biases, fp);
    CHK_READ(net->error_signals, sizeof(net->error_signals[0]), total_biases, fp);
    
    // Set initialization flag of network
    net->is_initialized = 1;
    
    fclose(fp);
    return NN_E_OK;
}

int nn_free(nn_arch *net)
{
    if (!net || !net->is_initialized)
        return NN_E_NET_UNINITIALIZED;
        
    net->n_hidden_layers = 0;
    net->is_initialized  = 0;
    
    NN_FREE_NULL(net->neurons);
    NN_FREE_NULL(net->offsets_neurons);
    NN_FREE_NULL(net->n_neurons);
    NN_FREE_NULL(net->weights);
    NN_FREE_NULL(net->biases);
    NN_FREE_NULL(net->error_signals);
    NN_FREE_NULL(net->offsets_weights);
    NN_FREE_NULL(net->offsets_biases);
    NN_FREE_NULL(net->activations);
    
    return NN_E_OK;
}

#endif /* NEURAL_NET_H */
