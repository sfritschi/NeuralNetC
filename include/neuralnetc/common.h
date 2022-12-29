#ifndef COMMON_H
#define COMMON_H

typedef float nn_scalar_t;
typedef nn_scalar_t (* nn_funcptr_t)(nn_scalar_t);

enum nn_errors {
    NN_E_OK = 0,
    NN_E_TOO_FEW_LAYERS,
    NN_E_OUT_OF_MEM
};

#define CHK_ALLOC(ptr) {\
    if (!(ptr)) {\
        return NN_E_OUT_OF_MEM;\
    }\
}

#endif /* COMMON_H */
