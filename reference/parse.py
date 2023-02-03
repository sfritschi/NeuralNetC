import sys
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.functional as F

activationNames   = ["identity", "sigmoid", "relu", "tanh"]
activationMapping = [None, nn.Sigmoid(), nn.ReLU(), nn.Tanh()]

class NetworkArch(nn.Module):
    
    def __init__(self, nLayers, nNeurons, activations):
        super(NetworkArch, self).__init__()
        
        # Verify that network is valid
        if (nLayers < 2 or len(nNeurons) != nLayers or len(activations) != nLayers - 1 \
            or not all(map(lambda x: x >= 1, nNeurons))):
            
            raise ValueError("Invalid arguments encountered")
            
        self.nLayers      = nLayers
        self.nNeurons     = nNeurons
        self.activations  = [activationMapping[a] for a in activations]
        
        self.layers = [nn.Linear(p, n) for p, n in \
                       zip(self.nNeurons[:-1], self.nNeurons[1:])]
        
    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if (activation is not None):
                x = activation(x)
        return x
    
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: parse.py <network.nnc>")
        exit(-1)
    
    # TODO: Put parsing of network file in constructor
    with open(sys.argv[1], "rb") as f:
        # Read binary little-endian data
        buf = f.read(3)
        x = unpack("<ccc", buf)
        if not (x[0] == b'N' and x[1] == b'N' and x[2] == b'C'):
            print("Wrong signature: Expected 'NNC'")
            exit(-1)
        
        buf = f.read(4)
        nLayers = unpack("<I", buf)[0]
        print(f"#Layers: {nLayers}")
        
        buf = f.read(4*nLayers)
        nNeurons = unpack("<" + "I" * nLayers, buf)
        print(f"#Neurons per Layer: {nNeurons}")
        
        buf = f.read(4*(nLayers - 1))
        activationCodes = unpack("<" + "i" * (nLayers - 1), buf)
        print(f"Activation functions: " +
            f"{[activationNames[c] if c is not None else activationNames[0] for c in activationCodes]}")
        
        totalNeurons = sum(nNeurons)
        totalWeights = sum([n * p for n, p in zip(nNeurons[1:], nNeurons[:-1])])
        totalBiases  = totalNeurons - nNeurons[0]
        
        print(f"Total #neurons: {totalNeurons}")
        print(f"Total #weights: {totalWeights}")
        print(f"Total #biases : {totalBiases}")

        # Read neurons
        buf = f.read(4*2*totalNeurons)  # * 2 since value + gradient (float)
        neurons = unpack("<" + "f" * 2 * totalNeurons, buf)
        #print(neurons)
        
        # Read weights
        buf = f.read(4*2*totalWeights)  # * 2 since value + gradient (float)
        weights = unpack("<" + "f" * 2 * totalWeights, buf)
        #print(weights)
        
        # Read biases
        buf = f.read(4*2*totalBiases)  # * 2 since value + gradient (float)
        biases = unpack("<" + "f" * 2 * totalBiases, buf)
        #print(biases)
    
    try: 
        net = NetworkArch(nLayers, nNeurons, activationCodes)
        
        #x = torch.zeros(6)
        #print(net(x))
        # Extract weights and biases (skipping over gradient fields)
        wt = torch.tensor([weights[i] for i in range(0, len(weights), 2)])
        bt = torch.tensor([biases[i] for i in range(0, len(biases), 2)])
        # Offsets for weights and biases
        ow = 0
        ob = 0
        # Set weights and biases
        for layer in net.layers:
            d_in  = layer.in_features
            d_out = layer.out_features
            
            pw = nn.parameter.Parameter(wt[ow:ow + d_out * d_in].reshape(d_out, d_in))
            pb = nn.parameter.Parameter(bt[ob:ob + d_out])
            
            params     = layer.parameters()
            param      = next(params)
            param.data = pw
            param      = next(params)
            param.data = pb
            
            # Update offsets
            ow += d_out * d_in
            ob += d_out
        
        # Compute forward pass
        x = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
        y_pred = net(x)
        
        print("Forward pass:")
        print(y_pred)
        
        # Compute backward pass
        y = torch.tensor([0.5, 1.0])
        loss = ((y - y_pred)**2).sum()
        loss.backward()
        
        # Print computed gradients of network
        print("Gradients:")
        for i, layer in enumerate(net.layers):
            print(f"Layer #{i+1}")
            for p in layer.parameters():
                print(p.grad)
            print("")
        
    except ValueError as e:
        print(str(e))
