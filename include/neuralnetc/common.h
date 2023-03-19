#ifndef NN_COMMON_H
#define NN_COMMON_H

#include <stdbool.h>
#include <endian.h>

typedef float nn_scalar_t;

static const char FILE_SIGNATURE[] = "NNC";
// Subtract 1 due to null terminator
#define SIGNATURE_LEN (sizeof(FILE_SIGNATURE) - 1)
// Numerically precision for "close to 0" comparisons
#define Epsilon 1e-8f

enum nn_errors {
    NN_E_OK = 0,
    NN_E_INVALID_DIMENSIONS,
    NN_E_NET_UNINITIALIZED,
    NN_E_NET_ALREADY_INITIALIZED,
    NN_E_TOO_FEW_LAYERS,
    NN_E_TOO_FEW_NEURONS,
    NN_E_OUT_OF_MEM,
    NN_E_FAILED_TO_WRITE_FILE,
    NN_E_FAILED_TO_READ_FILE,
    NN_E_UNRECOGNIZED_READ_SIGNATURE,
    NN_E_UNRECOGNIZED_ENUM_VALUE,
    NN_E_DATASET_INVALID,
    NN_E_DATASET_ALREADY_NORMALIZED,
    NN_E_DATASET_TRAIN_UNLABELLED,
    NN_E_DATASET_TEST_LABELLED,
    NN_E_NUMERIC_TOO_SMALL
};

#define CHK_ALLOC(ptr) do {\
    if (!(ptr)) {\
        return NN_E_OUT_OF_MEM;\
    }\
} while(0)

#define NN_FREE_NULL(ptr) do {\
    if ((ptr)) {\
        free(ptr);\
        ptr = NULL;\
    }\
} while(0)

#endif /* NN_COMMON_H */
