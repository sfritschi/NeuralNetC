#ifndef NN_RANDOM_H
#define NN_RANDOM_H

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <neuralnetc/pcg32.h>

#undef  M_PI
#define M_PI           3.14159265358979323846f

enum nn_param_init_type {
    NN_PARAM_INIT_PYTORCH = 0,    
    NN_PARAM_INIT_GLOROT    
};

nn_scalar_t random_uniform_scalar(pcg32 *gen, nn_scalar_t a, nn_scalar_t b)
{
    assert(a <= b && "Require a <= b!");
    
    const nn_scalar_t u = pcg32_next_scalar(gen);
    return (b - a) * u + a;
}

// Use Box-Muller transform to find two independent samples of normal distribution
nn_scalar_t random_normal_scalar(pcg32 *gen, nn_scalar_t mean, nn_scalar_t stddev)
{
    assert(stddev >= (nn_scalar_t)0.0 && "Require sigma to be positive!");
    
    static nn_scalar_t next = NAN;
    // Re-use second sample computed from previous iteration, if available
    if (!isnanf(next)) {
        const nn_scalar_t temp = next;
        next = NAN;
        return temp;
    }
    
    const nn_scalar_t u1 = pcg32_next_scalar(gen);
    const nn_scalar_t u2 = pcg32_next_scalar(gen);
    
    const nn_scalar_t r_ = stddev * sqrtf(-(nn_scalar_t)2.0 * logf(u1));
    const nn_scalar_t theta = (nn_scalar_t)2.0 * M_PI * u2;
    
    nn_scalar_t sin_theta, cos_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    
    next = r_*sin_theta + mean;
        
    return r_*cos_theta + mean;
}

#endif /* NN_RANDOM_H */
