#ifndef NN_RANDOM_H
#define NN_RANDOM_H

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <neuralnetc/pcg32.h>

#undef M_PI

#define M_PI           3.14159265358979323846f
#define INV_SQRT_TWO   0.70710678118654752440f
#define INV_SQRT_TWOPI 0.39894228040143267794f

nn_scalar_t random_uniform_scalar(pcg32 *gen, nn_scalar_t a, nn_scalar_t b)
{
    assert(a <= b && "Require a <= b!");
    
    const nn_scalar_t u = pcg32_next_scalar(gen);
    return (b - a) * u + a;
}

// Use Box-Muller transform to find two independent samples of normal distribution
nn_scalar_t random_normal_scalar(pcg32 *gen, nn_scalar_t mean, nn_scalar_t sigma)
{
    assert(sigma >= (nn_scalar_t)0.0 && "Require sigma to be positive!");
    
    static nn_scalar_t next = NAN;
    // Re-use second sample computed from previous iteration, if available
    if (!isnanf(next)) {
        const nn_scalar_t temp = next;
        next = NAN;
        return temp;
    }
    
    const nn_scalar_t u1 = pcg32_next_scalar(gen);
    const nn_scalar_t u2 = pcg32_next_scalar(gen);
    
    const nn_scalar_t r_ = sigma * sqrtf(-(nn_scalar_t)2.0 * logf(u1));
    const nn_scalar_t theta = (nn_scalar_t)2.0 * M_PI * u2;
    
    nn_scalar_t sin_theta, cos_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    
    next = r_*sin_theta + mean;
        
    return r_*cos_theta + mean;
}

/*
nn_scalar_t random_normal_scalar(pcg32 *gen, nn_scalar_t mean, nn_scalar_t stddev)
{
    // Newton's method parameters
    const uint32_t maxiter = 32;
    const nn_scalar_t rtol = (nn_scalar_t)1e-5;
    const nn_scalar_t atol = (nn_scalar_t)1e-6;
    
    // Generate uniform random sample
    const nn_scalar_t u = pcg32_next_scalar(gen);
    
    nn_scalar_t abs_dx;
    // PDF of normal distribution (initial guess)
    nn_scalar_t f  = INV_SQRT_TWOPI;
    // CDF of normal distribution (initial guess)
    nn_scalar_t F  = (nn_scalar_t)0.5;
    nn_scalar_t x  = (nn_scalar_t)0.0;  // initial guess (mean)
    nn_scalar_t dx = (nn_scalar_t)0.0;

    // Use Newton's method to iteratively find the inverse of the
    // standard normal CDF
    uint32_t i;
    for (i = 0; i < maxiter; ++i) {
        
        dx = (u - F) / f;
        x += dx;
        
        abs_dx = fabsf(dx);
        // Check for convergence of error
        if (abs_dx < atol || abs_dx < fabs(x) * rtol) {
            break;
        }
        
        // Evaluate PDF for next iteration
        f = INV_SQRT_TWOPI * expf(-(nn_scalar_t)0.5 * (x*x));
        // Evaluate CDF for next iteration
        F = (nn_scalar_t)0.5 * erfcf(-INV_SQRT_TWO * x);
    }
    
    // DEBUG:
    if (i == maxiter) {
        fprintf(stderr, "Max iterations reached!\n");
    }
    
    // Convert sample from standard normal distribution to arbitrary
    // normal distribution
    return stddev * x + mean;
}
*/

#endif /* NN_RANDOM_H */
