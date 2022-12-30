import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkArch(nn.Module):
    
    def __init__(self):
        super(NetworkArch, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 1)
        #self.fc = nn.Linear(1,1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc(x)
        return x

class Net2(nn.Module):
    
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(6, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 7)
        self.fc4 = nn.Linear(7, 5)
        self.fc5 = nn.Linear(5, 2)
        #self.fc = nn.Linear(1,1)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        #x = self.fc(x)
        return x
        
if __name__ == '__main__':
    
    net = Net2()
    
    x = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
    
    """
    params = net.parameters()
    with torch.no_grad():
        param = next(params)
        param.data = nn.parameter.Parameter(torch.tensor([[0.0]]))
        param = next(params)
        param.data = nn.parameter.Parameter(torch.tensor([0.0]))
    params = net.parameters()
    with torch.no_grad():
        param = next(params)
        param.data = nn.parameter.Parameter(torch.tensor([[0.0],[1.0]]))
        param = next(params)
        param.data = nn.parameter.Parameter(torch.tensor([0.0, 0.0]))
        param = next(params)
        param.data = nn.parameter.Parameter(torch.tensor([[2.0, 3.0]]))
        param = next(params)
        param.data = nn.parameter.Parameter(torch.tensor([0.0]))
    for p in net.parameters():
        print(p.data)
    """
    
    with torch.no_grad():
        for param in net.parameters():
            param.data = nn.parameter.Parameter(torch.ones(param.data.shape))
        
    y = torch.tensor([0.5, 1.0])
    
    y_pred = net(x)
    print("Prediction: {}".format(y_pred))
    
    loss = (y - y_pred)**2
    loss.sum().backward()
    
    print("Loss: {}".format(loss))
    
    print("Gradients")
    for p in net.parameters():
        print(p.grad)
    
