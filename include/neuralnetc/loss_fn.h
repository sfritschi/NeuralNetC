#ifndef NN_LOSS_FN_H
#define NN_LOSS_FN_H

#include <neuralnetc/common.h>

typedef nn_scalar_t (* nn_loss_funcptr_t)(nn_scalar_t, nn_scalar_t);

enum nn_loss_fn_type {
    NN_LOSS_SQUARED_ERROR = 0,
    NN_LOSS_CROSS_ENTROPY    
};

//static const char *NN_LOSS_FN_NAMES[] = {
//    "squared error",
//    "cross entropy"
//};

typedef struct {
    enum nn_loss_fn_type type;
    nn_loss_funcptr_t l;
    nn_loss_funcptr_t gradl;  
} nn_loss_fn;

nn_scalar_t nn_squared_error_func(nn_scalar_t f, nn_scalar_t y)
{
    const nn_scalar_t diff = f - y;
    return diff * diff;
}

nn_scalar_t nn_squared_error_grad(nn_scalar_t f, nn_scalar_t y)
{
    return (nn_scalar_t)2.0 * (f - y);
}

nn_scalar_t nn_cross_entropy_func(nn_scalar_t f, nn_scalar_t y)
{
    (void)y;
    return -logf(f);
}

nn_scalar_t nn_cross_entropy_grad(nn_scalar_t f, nn_scalar_t y)
{
    (void)y;
    return -(nn_scalar_t)1.0 / f;
}

static const nn_loss_fn NN_LOSS_FN[] = {
    {NN_LOSS_SQUARED_ERROR, &nn_squared_error_func, &nn_squared_error_grad},
    {NN_LOSS_CROSS_ENTROPY, &nn_cross_entropy_func, &nn_cross_entropy_grad}
};

#endif /* NN_LOSS_FN_H */
