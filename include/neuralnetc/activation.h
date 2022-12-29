#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include <neuralnetc/common.h>

nn_scalar_t nn_sigmoid(nn_scalar_t x)
{
    return (nn_scalar_t)1.0 / ((nn_scalar_t)1.0 + exp(-x));
}

nn_scalar_t nn_relu(nn_scalar_t x)
{
    return fmax((nn_scalar_t)0.0, x);
}

nn_scalar_t nn_tanh(nn_scalar_t x)
{
    return tanh(x);    
}

#endif /* ACTIVATION_H */
