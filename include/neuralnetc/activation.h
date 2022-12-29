#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include <neuralnetc/common.h>

enum nn_activation_type {
    NN_ACTIVATION_SIGMOID = 0,
    NN_ACTIVATION_RELU,
    NN_ACTIVATION_TANH
};

static const char *NN_ACTIVATION_NAMES[] = {
    "sigmoid",
    "ReLU",
    "tanh"
};

typedef struct {
    enum nn_activation_type type;
    nn_funcptr_t f;    
} nn_activation;

nn_scalar_t nn_sigmoid_func(nn_scalar_t x)
{
    return (nn_scalar_t)1.0 / ((nn_scalar_t)1.0 + exp(-x));
}

nn_scalar_t nn_ReLU_func(nn_scalar_t x)
{
    return fmax((nn_scalar_t)0.0, x);
}

nn_scalar_t nn_tanh_func(nn_scalar_t x)
{
    return tanh(x);    
}

// Declare global activation functions
static const nn_activation nn_sigmoid = {
    NN_ACTIVATION_SIGMOID,
    &nn_sigmoid_func
};

static const nn_activation nn_ReLU = {
    NN_ACTIVATION_RELU,
    &nn_ReLU_func
};

static const nn_activation nn_tanh = {
    NN_ACTIVATION_TANH,
    &nn_tanh_func
};

#endif /* ACTIVATION_H */
