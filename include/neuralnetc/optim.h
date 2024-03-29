#ifndef NN_OPTIM_H
#define NN_OPTIM_H

#include <neuralnetc/neuralnet.h>

int nn_optim_step_SGD(nn_arch *net, const nn_dataset *train, 
                      nn_scalar_t learning_rate, nn_scalar_t weight_decay,
                      enum nn_loss_fn_type loss_type, nn_scalar_t *train_loss)
{
    assert(weight_decay >= 0.0 && "Require non-negative weight_decay");
    assert(net && train && "Expected non-NULL pointers");
    
    if (train->labels == NULL)
        return NN_E_DATASET_TRAIN_UNLABELLED;
    
    if (net->n_neurons[0] != train->sample_dim ||
        net->n_neurons[net->n_hidden_layers+1] != train->label_dim)
        return NN_E_INVALID_DIMENSIONS;
    
    // TODO: Keep track of training loss/validation loss history
    const nn_loss_funcptr_t loss_fn = NN_LOSS_FN[loss_type].l;
    
    const uint32_t total_weights = net->offsets_weights[net->n_hidden_layers+1];
    const uint32_t total_biases  = net->offsets_biases[net->n_hidden_layers+1];
    // Offset of neurons in output layer
    const uint32_t on_output = net->offsets_neurons[net->n_hidden_layers+1];
    
    uint32_t i, j, k, local_batch, start = 0;
    nn_scalar_t train_l = 0.0;
    for (i = 0; i < train->n_batches; ++i) {
        // Set gradients to 0
        for (j = 0; j < total_weights; ++j) net->weights[j].grad = 0.0;
        for (j = 0; j < total_biases; ++j)  net->biases[j].grad  = 0.0;
        
        local_batch = nn_dataset_local_batch_size(train, i);
        
        // TODO: Account for different losses
        // Compute (average) batch loss
        for (j = 0; j < local_batch; ++j) {
            const nn_scalar_t *sample = &train->data[FLATTEN(j+start,0,train->sample_dim)];
            
            // Compute Forward pass
            nn_forward(net, sample);
            
            const nn_scalar_t *y_label = &train->labels[FLATTEN(j+start,0,train->label_dim)];
            
            nn_scalar_t f, y;
            for (k = 0; k < train->label_dim; ++k) {
                f = net->neurons[k + on_output].value;
                y = y_label[k];
                train_l += loss_fn(f, y);
            }
            
            // Compute Backward pass (accumulate gradients)
            nn_backward(net, y_label, loss_type);
        }
        train_l /= (nn_scalar_t)local_batch;
        
        // Update weights and biases based on accumulated loss gradients (averaged)
        // plus (optional) weight decay
        if (weight_decay == (nn_scalar_t)0.0)
            for (j = 0; j < total_weights; ++j)
                net->weights[j].value -= learning_rate * net->weights[j].grad / (nn_scalar_t)local_batch;
        else
            for (j = 0; j < total_weights; ++j)
                net->weights[j].value -= learning_rate * 
                    (net->weights[j].grad / (nn_scalar_t)local_batch +
                     2.0 * weight_decay * net->weights[j].value);
        
        for (j = 0; j < total_biases; ++j)
            net->biases[j].value -= learning_rate * net->biases[j].grad / (nn_scalar_t)local_batch;
        
        start += local_batch;
    }
    train_l /= (nn_scalar_t)train->n_batches;
    // Set computed training loss
    if (train_loss) *train_loss = train_l;
    
    return NN_E_OK;
}

#endif /* NN_OPTIM_H */
