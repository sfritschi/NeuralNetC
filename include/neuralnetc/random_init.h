#ifndef NN_RANDOM_H
#define NN_RANDOM_H

#include <stdio.h>
#include <math.h>

#include <neuralnetc/pcg32.h>

#define INV_SQRT_TWO   0.70710678118654752440f
#define INV_SQRT_TWOPI 0.39894228040143267794f

nn_scalar_t uniform_scalar_range(pcg32 *gen, nn_scalar_t a, nn_scalar_t b)
{
    const nn_scalar_t u = pcg32_next_scalar(gen);
    return (b - a) * u + a;
}

nn_scalar_t normal_scalar(pcg32 *gen, nn_scalar_t mean, nn_scalar_t stddev)
{
    // Newton's method parameters
    const uint32_t maxiter = 32;
    const nn_scalar_t rtol = (nn_scalar_t)1e-5;
    const nn_scalar_t atol = (nn_scalar_t)1e-6;
    
    const nn_scalar_t inv_stddev = (nn_scalar_t)1.0 / stddev;
    const nn_scalar_t pdf_norm = INV_SQRT_TWOPI * inv_stddev;
    const nn_scalar_t pdf_scale = (nn_scalar_t)0.5 * inv_stddev*inv_stddev;
    const nn_scalar_t scale = INV_SQRT_TWO * inv_stddev;
    // Generate uniform random sample
    const nn_scalar_t u = pcg32_next_scalar(gen);
    
    nn_scalar_t diff;  // difference mean - x
    // PDF of normal distribution (initial guess)
    nn_scalar_t f  = pdf_norm;
    // CDF of normal distribution (initial guess)
    nn_scalar_t F  = (nn_scalar_t)0.5;
    nn_scalar_t x  = mean;  // initial guess
    nn_scalar_t dx = (nn_scalar_t)0.0;

    uint32_t i;
    for (i = 0; i < maxiter; ++i) {
        
        dx = (F - u) / f;
        x -= dx;
        
        // Check for convergence of error
        if (fabs(dx) < atol || fabs(dx) < fabs(x) * rtol) {
            break;
        }
        
        diff = mean - x;
        // Evaluate PDF for next iteration
        f = pdf_norm * exp(-pdf_scale * (diff*diff));
        // Evaluate CDF for next iteration
        F = (nn_scalar_t)0.5 * erfc(scale * diff);
    }
    
    // DEBUG:
    if (i == maxiter) {
        fprintf(stderr, "Max iterations reached!\n");
    }
    
    printf("#Needed iterations: %u\n", i+1);
    return x;
}

#endif /* NN_RANDOM_H */
