/*
 * Tiny self-contained version of the PCG Random Number Generation for C++
 * put together from pieces of the much larger C/C++ codebase.
 * Wenzel Jakob, February 2015
 *
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#ifndef __PCG32_H
#define __PCG32_H

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

#include <stdint.h>

#include <neuralnetc/common.h>  // nn_scalar_t

/// PCG32 Pseudorandom number generator
typedef struct {
    uint64_t state;  // RNG state.  All values are possible.
    uint64_t inc;    // Controls which RNG sequence (stream) is selected. Must *always* be odd.
} pcg32;

pcg32 pcg32_init()
{
    return (pcg32) {PCG32_DEFAULT_STATE, PCG32_DEFAULT_STREAM};
}

/// Generate a uniformly distributed unsigned 32-bit random number
uint32_t pcg32_next_uint(pcg32 *gen)
{
    const uint64_t oldstate = gen->state;
    gen->state = oldstate * PCG32_MULT + gen->inc;
    const uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
    const uint32_t rot = (uint32_t) (oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

/**
* \brief Seed the pseudorandom number generator
*
* Specified in two parts: a state initializer and a sequence selection
* constant (a.k.a. stream id)
*/
void pcg32_seed(pcg32 *gen, uint64_t initstate)
{
    const uint64_t initseq = 1U;  // default stream id
    
    gen->state = 0U;
    gen->inc = (initseq << 1u) | 1u;
    pcg32_next_uint(gen);
    gen->state += initstate;
    pcg32_next_uint(gen);
}

/// Generate a uniformly distributed number, r, where 0 <= r < bound
uint32_t pcg32_next_uint_bound(pcg32 *gen, uint32_t bound)
{
    // To avoid bias, we need to make the range of the RNG a multiple of
    // bound, which we do by dropping output less than a threshold.
    // A naive scheme to calculate the threshold would be to do
    //
    //     uint32_t threshold = 0x100000000ull % bound;
    //
    // but 64-bit div/mod is slower than 32-bit div/mod (especially on
    // 32-bit platforms).  In essence, we do
    //
    //     uint32_t threshold = (0x100000000ull-bound) % bound;
    //
    // because this version will calculate the same modulus, but the LHS
    // value is less than 2^32.

    const uint32_t threshold = (~bound+1u) % bound;

    // Uniformity guarantees that this loop will terminate.  In practice, it
    // should usually terminate quickly; on average (assuming all bounds are
    // equally likely), 82.25% of the time, we can expect it to require just
    // one iteration.  In the worst case, someone passes a bound of 2^31 + 1
    // (i.e., 2147483649), which invalidates almost 50% of the range.  In
    // practice, bounds are typically small and only a tiny amount of the range
    // is eliminated.
    for (;;) {
        const uint32_t r = pcg32_next_uint(gen);
        if (r >= threshold)
            return r % bound;
    }
}

/// Generate a single precision floating point value on the interval [0, 1)
nn_scalar_t pcg32_next_scalar(pcg32 *gen) 
{
    /* Trick from MTGP: generate an uniformly distributed
        single precision number in [1,2) and subtract 1. */
    union {
        uint32_t u;
        nn_scalar_t s;
    } x;
    x.u = (pcg32_next_uint(gen) >> 9) | 0x3f800000u;
    return x.s - 1.0f;
}

#endif // __PCG32_H
