#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include <neuralnetc/common.h>

enum nn_activation_type {
    NN_ACTIVATION_IDENTITY = 0,
    NN_ACTIVATION_SIGMOID,
    NN_ACTIVATION_RELU,
    NN_ACTIVATION_TANH
};

static const char *NN_ACTIVATION_NAMES[] = {
    "identity",
    "sigmoid",
    "ReLU",
    "tanh"
};

typedef struct {
    enum nn_activation_type type;
    nn_funcptr_t f;
    nn_funcptr_t gradf;  
} nn_activation;

static nn_scalar_t nn_sigmoid_func(nn_scalar_t x)
{
    return (nn_scalar_t)1.0 / ((nn_scalar_t)1.0 + exp(-x));
}

static nn_scalar_t nn_sigmoid_grad(nn_scalar_t a)
{
    return a * ((nn_scalar_t)1.0 - a);
}

static nn_scalar_t nn_ReLU_func(nn_scalar_t x)
{
    return fmax((nn_scalar_t)0.0, x);
}

static nn_scalar_t nn_ReLU_grad(nn_scalar_t a)
{
    return (a > (nn_scalar_t)0.0) ? (nn_scalar_t)1.0 : (nn_scalar_t)0.0;    
}

static nn_scalar_t nn_tanh_func(nn_scalar_t x)
{
    return tanh(x);    
}

static nn_scalar_t nn_tanh_grad(nn_scalar_t a)
{
    return (nn_scalar_t)1.0 - a*a;
}

// Declare global activation functions
static const nn_activation nn_identity = {
    NN_ACTIVATION_IDENTITY,
    NULL,
    NULL
};

static const nn_activation nn_sigmoid = {
    NN_ACTIVATION_SIGMOID,
    &nn_sigmoid_func,
    &nn_sigmoid_grad
};

static const nn_activation nn_ReLU = {
    NN_ACTIVATION_RELU,
    &nn_ReLU_func,
    &nn_ReLU_grad
};

static const nn_activation nn_tanh = {
    NN_ACTIVATION_TANH,
    &nn_tanh_func,
    &nn_tanh_grad
};

#endif /* ACTIVATION_H */
