import sys
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.functional as F

activationNames   = ["identity", "sigmoid", "relu", "tanh"]
activationMapping = [None, nn.Sigmoid(), nn.ReLU(), nn.Tanh()]

class NetworkArch(nn.Module):
    
    def __init__(self, filename, has_output=False):
        super(NetworkArch, self).__init__()
        
        with open(filename, "rb") as f:
            # Read binary little-endian data
            buf = f.read(3)
            x = unpack("<ccc", buf)
            # Check for correct file signature
            if not (x[0] == b'N' and x[1] == b'N' and x[2] == b'C'):
                print("Wrong signature: Expected 'NNC'")
                raise RuntimeError("File does not have expected signature 'NNC'")
            
            buf = f.read(4)
            self.nLayers = unpack("<I", buf)[0]
            # Make sure at least 2 layers (input & output)
            if (self.nLayers < 2):
                raise RuntimeError("Expected at least 2 layers (input & output)")
                
            print(f"#Layers: {self.nLayers}")
            
            buf = f.read(4*self.nLayers)
            self.nNeurons = unpack("<" + "I" * self.nLayers, buf)
            # Make sure number of neurons positive in every layer
            if (not all(map(lambda x: x >= 1, self.nNeurons))):
                raise RuntimeError("Require at least 1 neuron in every layer")
                
            print(f"#Neurons per Layer: {self.nNeurons}")
            self.layers = [nn.Linear(p, n) for p, n in \
                           zip(self.nNeurons[:-1], self.nNeurons[1:])]
                       
            buf = f.read(4*(self.nLayers - 1))
            activations = unpack("<" + "i" * (self.nLayers - 1), buf)
            print(f"Activation functions: " +
                f"{[activationNames[c] if c is not None else activationNames[0] for c in activations]}")
            self.activations = [activationMapping[a] for a in activations]
            
            totalNeurons = sum(self.nNeurons)
            totalWeights = sum([n * p for n, p in zip(self.nNeurons[1:], self.nNeurons[:-1])])
            totalBiases  = totalNeurons - self.nNeurons[0]
            
            print(f"Total #neurons: {totalNeurons}")
            print(f"Total #weights: {totalWeights}")
            print(f"Total #biases : {totalBiases}")
    
            # Read neurons (not needed)
            buf = f.read(4*2*totalNeurons)  # * 2 since value + gradient (float)
            if has_output:
                neurons = unpack("<" + "f" * 2 * totalNeurons, buf)
                # Extract output from forward pass (neurons in last layer,
                # where we skip every other gradient value)
                self.output = torch.tensor(neurons[-2*self.nNeurons[-1]::2])
                
            # Read weights
            buf = f.read(4*2*totalWeights)  # * 2 since value + gradient (float)
            weights = unpack("<" + "f" * 2 * totalWeights, buf)
            
            # Read biases
            buf = f.read(4*2*totalBiases)  # * 2 since value + gradient (float)
            biases = unpack("<" + "f" * 2 * totalBiases, buf)
        
        # Explicitly set weights and biases from file
        
        # Extract weights and biases (including gradients)
        wt  = torch.tensor([weights[i] for i in range(0, len(weights), 2)])
        gwt = torch.tensor([weights[i] for i in range(1, len(weights), 2)])
        bt  = torch.tensor([biases[i] for i in range(0, len(biases), 2)])
        gbt = torch.tensor([biases[i] for i in range(1, len(biases), 2)])
        # Offsets for weights and biases
        ow = 0
        ob = 0
        # Set weights and biases
        for layer in self.layers:
            d_in  = layer.in_features
            d_out = layer.out_features
            
            pw = nn.parameter.Parameter(wt[ow:ow + d_out * d_in].reshape(d_out, d_in))
            gw = nn.parameter.Parameter(gwt[ow:ow + d_out * d_in].reshape(d_out, d_in))
            pb = nn.parameter.Parameter(bt[ob:ob + d_out])
            gb = nn.parameter.Parameter(gbt[ob:ob + d_out])
            
            # Note: Assumes that parameters consist of (weights, biases) pairs
            params     = layer.parameters()
            param      = next(params)
            # Weights
            param.data = pw
            param.grad = gw
            param      = next(params)
            # Biases
            param.data = pb
            param.grad = gb
            
            # Update offsets
            ow += d_out * d_in
            ob += d_out
        
    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if (activation is not None):
                x = activation(x)
        return x
    
if __name__ == '__main__':
    
    try: 
        net_init = NetworkArch("net_initial.nnc")
        net_back = NetworkArch("net_backward.nnc", has_output=True)
        
        # TODO: Replace hard-coded network input (x) and output labels (y)
        
        # Compute forward pass
        x = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
        
        y_pred = net_init(x)
        
        print("")
        print("Expected forward pass:")
        print(y_pred)
        print("Forward pass from NeuralNetC:")
        print(net_back.output)
        print("")
        rmse = torch.mean((y_pred - net_back.output)**2)**0.5
        print(f"RMSE forward pass: {rmse.item()}")
        
        # Compute backward pass
        y = torch.tensor([0.5, 1.0])
        
        loss = ((y - y_pred)**2).sum()
        loss.backward()
        
        # Compute errors/differences in computed gradients
        rmse_grad_weights = []
        rmse_grad_biases  = []
    
        for layer_init, layer_back in zip(net_init.layers, net_back.layers):
            for pi, pb in zip(layer_init.parameters(), layer_back.parameters()):
                rmse = torch.mean((pi.grad - pb.grad)**2)**0.5
                if (len(pi.data.shape) == 1):
                    rmse_grad_biases.append(rmse.item())
                else:
                    rmse_grad_weights.append(rmse.item())
        
        N = len(rmse_grad_weights)
        assert(len(rmse_grad_biases) == N)
        
        # Print min., mean and max. RMSE for all gradients across all layers
        print(f"RMSE grad. weights: Min: {min(rmse_grad_weights)} Mean: {sum(rmse_grad_weights)/N} Max: {max(rmse_grad_weights)}")
        print(f"RMSE grad. biases : Min: {min(rmse_grad_biases)} Mean: {sum(rmse_grad_biases)/N} Max: {max(rmse_grad_biases)}")
        
    except RuntimeError as e:
        print(str(e))
