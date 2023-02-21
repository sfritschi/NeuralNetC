#ifndef NN_DATASET_H
#define NN_DATASET_H

#include <stdio.h>
#include <assert.h>

#include <neuralnetc/common.h>
#include <neuralnetc/random_init.h>

#define FLATTEN(i, j, cols) ((i) * (cols) + (j))

enum nn_dataset_norm_type {
    NN_DATASET_UNNORMALIZED = 0,
    NN_DATASET_NORMALIZED_MIN_MAX,
    NN_DATASET_NORMALIZED_STANDARD
};

typedef struct {
    nn_scalar_t *data;    // data is stored in row-major (C) order
    nn_scalar_t *labels;  // (optional) per sample label
    uint32_t n_samples;
    uint32_t sample_dim;
    uint32_t label_dim;
    uint32_t n_batches;
    uint32_t batch_size;
    // buffers to store parameters of (optional) normalization
    nn_scalar_t *__normalize_bufferA; 
    nn_scalar_t *__normalize_bufferB;
    uint32_t __remainder;  // Keep track of remainder(n_samples, batch_size)
    enum nn_dataset_norm_type normalization;
} nn_dataset;

int nn_dataset_init_unlabelled(nn_dataset *dataset, uint32_t n_samples, 
                               uint32_t sample_dim, uint32_t batch_size)
{
    assert(dataset && "Expected non-NULL dataset");
    
    if (n_samples < 1 || sample_dim < 1 || batch_size < 1 || n_samples < batch_size)
        return NN_E_DATASET_INVALID;
    
    CHK_ALLOC(dataset->data = (nn_scalar_t *) malloc(n_samples * sample_dim * sizeof(nn_scalar_t)));
    dataset->labels = NULL;
    
    // set normalization buffers to NULL initially
    dataset->__normalize_bufferA = NULL;
    dataset->__normalize_bufferB = NULL;
    dataset->normalization = NN_DATASET_UNNORMALIZED;
    
    dataset->n_samples  = n_samples;
    dataset->sample_dim = sample_dim;
    dataset->label_dim  = 0;
    // Compute ceiling of #samples divided by batch_size
    // NOTE: This could theoretically result in overflow of numerator
    dataset->n_batches   = (n_samples + batch_size - 1) / batch_size;
    dataset->__remainder = n_samples % batch_size;
    dataset->batch_size  = batch_size;
    
    return NN_E_OK;
}

int nn_dataset_init_labelled(nn_dataset *dataset, uint32_t n_samples, 
                             uint32_t sample_dim, uint32_t batch_size, 
                             uint32_t label_dim)
{
    assert(dataset && "Expected non-NULL dataset");
    
    if (n_samples < 1 || sample_dim < 1 || label_dim < 1 || batch_size < 1 || n_samples < batch_size)
        return NN_E_DATASET_INVALID;
    
    CHK_ALLOC(dataset->data = (nn_scalar_t *) malloc(n_samples * sample_dim * sizeof(nn_scalar_t)));
    CHK_ALLOC(dataset->labels = (nn_scalar_t *) malloc(n_samples * label_dim * sizeof(nn_scalar_t)));
    
    // set normalization buffers to NULL initially
    dataset->__normalize_bufferA = NULL;
    dataset->__normalize_bufferB = NULL;
    dataset->normalization = NN_DATASET_UNNORMALIZED;
    
    dataset->n_samples  = n_samples;
    dataset->sample_dim = sample_dim;
    dataset->label_dim  = label_dim;
    // Compute ceiling of #samples divided by batch_size
    // NOTE: This could theoretically result in overflow of numerator
    dataset->n_batches   = (n_samples + batch_size - 1) / batch_size;
    dataset->__remainder = n_samples % batch_size;
    dataset->batch_size  = batch_size;
    
    return NN_E_OK;
}

uint32_t nn_dataset_local_batch_size(const nn_dataset *dataset, uint32_t batch_index)
{
    assert(dataset && batch_index < dataset->n_batches && "Expected non-NULL dataset");
    
    // Last batch contains remainder of samples if the total #samples
    // is not evenly divisible by batch_size
    if (dataset->__remainder != 0 && batch_index + 1 == dataset->n_batches)
        return dataset->__remainder;
    else
        return dataset->batch_size;
}

void nn_dataset_random_shuffle_samples(pcg32 *gen, nn_dataset *dataset)
{
    assert(dataset && "Expected non-NULL dataset");
    
    const uint32_t N = dataset->n_samples;
    const uint32_t d = dataset->sample_dim;
    const uint32_t l = dataset->label_dim;
    
    nn_scalar_t temp;
    // NOTE: Assumes n_samples >= 1, which is true if initialized properly
    // Fisher-Yates random shuffle algorithm
    uint32_t i, j, from, to;
    for (i = 0; i < N - 1; ++i) {
        // Random index in range [i, N) 
        const uint32_t random_index = i + pcg32_next_uint_bound(gen, N - i);
        // Swap samples
        for (j = 0; j < d; ++j) {
            from = FLATTEN(i, j, d);
            to   = FLATTEN(random_index, j, d);
            
            temp                = dataset->data[to];
            dataset->data[to]   = dataset->data[from];
            dataset->data[from] = temp;
        }
        
        // Swap also associated labels if available
        if (dataset->labels) {
            for (j = 0; j < l; ++j) {
                from = FLATTEN(i, j, l);
                to   = FLATTEN(random_index, j, l);
                
                temp                  = dataset->labels[to];
                dataset->labels[to]   = dataset->labels[from];
                dataset->labels[from] = temp;
            }
        }
    }
}

int nn_dataset_min_max(nn_dataset *dataset, bool transform_only)
{
    assert(dataset && "Expected non-NULL dataset");
    
    uint32_t i, j;
    nn_scalar_t val;
    
    if (!transform_only) {
        // Initialization
        for (j = 0; j < dataset->sample_dim; ++j) {
            dataset->__normalize_bufferA[j] = INFINITY;
            dataset->__normalize_bufferB[j] = -INFINITY;
        }
        
        // Compute minimum/maximum values for each dimension
        for (i = 0; i < dataset->n_samples; ++i) {
            for (j = 0; j < dataset->sample_dim; ++j) {
                val = dataset->data[FLATTEN(i, j, dataset->sample_dim)];
                // bufferA/B represents minimum/maximum value for each dim.
                dataset->__normalize_bufferA[j] = fminf(dataset->__normalize_bufferA[j], val);
                dataset->__normalize_bufferB[j] = fmaxf(dataset->__normalize_bufferB[j], val);
            }
        }
        
        // Post-processing; set bufferB to be xmax - xmin in each dimension
        for (j = 0; j < dataset->sample_dim; ++j) {
            dataset->__normalize_bufferB[j] -= dataset->__normalize_bufferA[j];
            if (fabsf(dataset->__normalize_bufferB[j]) < Epsilon) {
                return NN_E_NUMERIC_TOO_SMALL;
            }
        }
    }
    
    // Apply transformation to dataset
    for (i = 0; i < dataset->n_samples; ++i) {
        for (j = 0; j < dataset->sample_dim; ++j) {
            const uint32_t index = FLATTEN(i, j, dataset->sample_dim);
            val = dataset->data[index];
            // val <- (val - xmin) / (xmax - xmin)
            val = (val - dataset->__normalize_bufferA[j]) / dataset->__normalize_bufferB[j];
            dataset->data[index] = val;
        }
    }
    
    return NN_E_OK;
}

int nn_dataset_standardize(nn_dataset *dataset, bool transform_only)
{
    assert(dataset && "Expected non-NULL dataset");
    
    uint32_t i, j;
    nn_scalar_t val;
    
    if (!transform_only) {
        // Initialization
        for (j = 0; j < dataset->sample_dim; ++j) {
            dataset->__normalize_bufferA[j] = 0.0;
            dataset->__normalize_bufferB[j] = 0.0;
        }
        
        // Compute mean/stddev values for each dimension
        for (i = 0; i < dataset->n_samples; ++i) {
            for (j = 0; j < dataset->sample_dim; ++j) {
                val = dataset->data[FLATTEN(i, j, dataset->sample_dim)];
                // bufferA/B represents sum/sum of squares for each dim.
                dataset->__normalize_bufferA[j] += val;
                dataset->__normalize_bufferB[j] += val*val;
            }
        }
        
        const nn_scalar_t N = (nn_scalar_t)dataset->n_samples;
        // Post-processing; Set bufferA to mean and bufferB to stddev
        for (j = 0; j < dataset->sample_dim; ++j) {
            const nn_scalar_t mean = dataset->__normalize_bufferA[j] / N;
            dataset->__normalize_bufferA[j] = mean;
            dataset->__normalize_bufferB[j] = sqrtf(dataset->__normalize_bufferB[j] / N - mean*mean);
            
            if (fabsf(dataset->__normalize_bufferB[j]) < Epsilon) {
                return NN_E_NUMERIC_TOO_SMALL;
            }
        }
    }
    
    // Apply transformation to dataset
    for (i = 0; i < dataset->n_samples; ++i) {
        for (j = 0; j < dataset->sample_dim; ++j) {
            const uint32_t index = FLATTEN(i, j, dataset->sample_dim);
            val = dataset->data[index];
            // val <- (val - mean) / stddev
            val = (val - dataset->__normalize_bufferA[j]) / dataset->__normalize_bufferB[j];
            dataset->data[index] = val;
        }
    }
    
    return NN_E_OK;
}

int nn_dataset_normalize(nn_dataset *dataset, enum nn_dataset_norm_type norm)
{
    assert(dataset && "Expected non-NULL dataset");
    
    // ?TODO: Verify that normalization type is actually valid
    
    // Currently, only support 1 type of transformation per dataset
    if (dataset->normalization != NN_DATASET_UNNORMALIZED)
        return NN_E_DATASET_ALREADY_NORMALIZED;
    
    // Allocate auxiliary buffers needed to keep track of transformation
    CHK_ALLOC(dataset->__normalize_bufferA = (nn_scalar_t *) malloc(dataset->sample_dim * sizeof(nn_scalar_t)));
    CHK_ALLOC(dataset->__normalize_bufferB = (nn_scalar_t *) malloc(dataset->sample_dim * sizeof(nn_scalar_t)));
    
    dataset->normalization = norm;
    
    enum nn_errors err = NN_E_OK;
    
    switch (norm) {
        case NN_DATASET_NORMALIZED_MIN_MAX:
            err = nn_dataset_min_max(dataset, false);
            break;
        
        case NN_DATASET_NORMALIZED_STANDARD:
            err = nn_dataset_standardize(dataset, false);
            break;
        
        default:
            assert(false && "Unreachable");
    }
    
    return err;
}

int nn_dataset_transfer_normalization(const nn_dataset *from, nn_dataset *to)
{
    assert(from && to && "Expected non-NULL datasets");
    
    if (to->normalization != NN_DATASET_UNNORMALIZED)
        return NN_E_DATASET_ALREADY_NORMALIZED;
    
    if (to->sample_dim != from->sample_dim)
        return NN_E_DATASET_INVALID;
        
    if (from->normalization == NN_DATASET_UNNORMALIZED)
        return NN_E_OK;  // nothing to do
    
    // Allocate auxiliary buffers needed to keep track of transformation
    CHK_ALLOC(to->__normalize_bufferA = (nn_scalar_t *) malloc(to->sample_dim * sizeof(nn_scalar_t)));
    CHK_ALLOC(to->__normalize_bufferB = (nn_scalar_t *) malloc(to->sample_dim * sizeof(nn_scalar_t)));
    // Copy transformation values of 'from' to 'to'
    for (uint32_t j = 0; j < from->sample_dim; ++j) {
        to->__normalize_bufferA[j] = from->__normalize_bufferA[j];
        to->__normalize_bufferB[j] = from->__normalize_bufferB[j];
    }
    to->normalization = from->normalization;
    
    enum nn_errors err = NN_E_OK;
    
    switch (from->normalization) {
        case NN_DATASET_NORMALIZED_MIN_MAX:
            err = nn_dataset_min_max(to, true);
            break;
        
        case NN_DATASET_NORMALIZED_STANDARD:
            err = nn_dataset_standardize(to, true);
            break;
        
        default:
            assert(false && "Unreachable");
    }
    
    return err;
}

// DEBUG
void nn_dataset_fill_random(pcg32 *gen, nn_dataset *dataset, nn_scalar_t low, nn_scalar_t high)
{
    assert(dataset && "Expected non-NULL dataset");
    
    for (uint32_t i = 0; i < dataset->n_samples; ++i) {
        for (uint32_t j = 0; j < dataset->sample_dim; ++j) {
            const nn_scalar_t random = random_uniform_scalar(gen, low, high);
            dataset->data[FLATTEN(i, j, dataset->sample_dim)] = random;
        }
    }
}

void nn_dataset_fill_continuous(nn_dataset *dataset)
{
    assert(dataset && "Expected non-NULL dataset");
    
    for (uint32_t i = 0; i < dataset->n_samples; ++i) {
        for (uint32_t j = 0; j < dataset->sample_dim; ++j) {
            const uint32_t index = FLATTEN(i, j, dataset->sample_dim);
            dataset->data[index] = (nn_scalar_t)index;
        }
    }
}

void nn_dataset_print(const nn_dataset *dataset)
{
    assert(dataset && "Expected non-NULL dataset");
    
    uint32_t i, j;
    for (i = 0; i < dataset->n_samples; ++i) {
        for (j = 0; j < dataset->sample_dim; ++j) {
            printf("%.8f ", dataset->data[FLATTEN(i, j, dataset->sample_dim)]);
        }
        
        if (dataset->labels) {
            printf("\ty = ");
            for (j = 0; j < dataset->label_dim; ++j) {
                printf("%.8f ", dataset->labels[FLATTEN(i, j, dataset->label_dim)]);
            }
        }
        printf("\n");
    }
}

// TODO: Read dataset from file (labelled or unlabelled? Batch size?)
int nn_dataset_read(nn_dataset *dataset, const char *filename) 
{
    assert(dataset && "Expected non-NULL dataset");
    
    FILE *fp = fopen(filename, "r");
    if (!fp) return NN_E_FAILED_TO_READ_FILE;
    
    /*
    uint32_t i, j;
    for (i = 0; i < dataset->n_samples; ++i) {
        for (j = 0; j < dataset->sample_dim; ++j) {
            fscanf(fp, "%f", &dataset->data[FLATTEN(i, j, dataset->sample_dim)]);
        }
        
        if (dataset->labels) {
            for (j = 0; j < dataset->label_dim; ++j) {
                fscanf(fp, "%f", &dataset->labels[FLATTEN(i, j, dataset->label_dim)]);
            }
        }
    }
    */
    
    fclose(fp);
    return NN_E_OK;
}

int nn_dataset_write(const nn_dataset *dataset, const char *filename)
{
    assert(dataset && "Expected non-NULL dataset");
    
    FILE *fp = fopen(filename, "w");
    if (!fp) return NN_E_FAILED_TO_WRITE_FILE;
    
    // Note: Labels (if any) are written in last columns
    uint32_t i, j;
    for (i = 0; i < dataset->n_samples; ++i) {
        for (j = 0; j < dataset->sample_dim; ++j) {
            fprintf(fp, "%.8f ", dataset->data[FLATTEN(i, j, dataset->sample_dim)]);
        }
        
        if (dataset->labels) {
            for (j = 0; j < dataset->label_dim; ++j) {
                fprintf(fp, "%.8f ", dataset->labels[FLATTEN(i, j, dataset->label_dim)]);
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    return NN_E_OK;
}

int nn_dataset_free(nn_dataset *dataset)
{
    assert(dataset && "Expected non-NULL dataset");
    
    NN_FREE_NULL(dataset->data);
    NN_FREE_NULL(dataset->labels);
    NN_FREE_NULL(dataset->__normalize_bufferA);
    NN_FREE_NULL(dataset->__normalize_bufferB);
    
    dataset->n_samples   = 0;
    dataset->sample_dim  = 0;
    dataset->label_dim   = 0;
    dataset->n_batches   = 0;
    dataset->__remainder = 0;
    dataset->batch_size  = 0;
    
    dataset->normalization = NN_DATASET_UNNORMALIZED;
    
    return NN_E_OK;
}

#endif /* NN_DATASET_H */
