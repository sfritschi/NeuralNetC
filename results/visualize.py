import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    train = np.loadtxt("train.dat")
    test  = np.loadtxt("test.dat")
    
    # Undo min-max scaling
    train[:,0] *= 2.0 * np.pi
    test[:,0]  *= 2.0 * np.pi
    
    gt = np.copy(test)
    gt[:,1] = np.sin(gt[:,0])
    
    plt.figure()
    plt.title(r"NN for dataset $y = \sin(x) + \epsilon,\;\epsilon\sim\mathcal{N}(0, \sigma^2)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$", rotation=0)
    #plt.plot(train[:,0], train[:,1], "ob", label=r"training data")
    plt.plot(gt[:,0], gt[:,1], "--k", label=r"ground truth")
    plt.plot(test[:,0], test[:,1], "-r", label=r"predicted")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot.png")
    plt.close()
