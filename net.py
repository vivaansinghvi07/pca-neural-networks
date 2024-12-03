import torch
from torch import nn
from data import *
from sklearn.metrics import confusion_matrix



class NormalNetwork(nn.Module):
    """
    Input shape: (N, 728)  -- the flattened images
    Output shape: (N, 10)  -- the class probabilities
    """
    def __init__(self, batch_size = 32):
        # normalNetWork inherits nn.Module's attributes
        super(NormalNetwork, self).__init__()
        self.forward_prob = nn.Sequential(
        # input is a batch of 28x28 images, by default - weights are initialized randomly between [ -sqrt(1/784), sqrt(1/784) ] and biases are initialized to zero
            nn.Linear(pow(28, 2), 512), 
            nn.ReLU(), # hidden layer 1, relu => max(0, x)
            nn.Linear(512, 512), 
            nn.ReLU(), # hidden layer 2
            nn.Linear(512, 10), 
        )
        self.train_loader = get_mnist_raw(train=True, batch_size= batch_size)
        self.test_loader = get_mnist_raw(train=False, batch_size = batch_size)
        # applies LogSoftmax then NLLLoss
        self.lossFunc = nn.CrossEntropyLoss()

    def forward(self, input_x):
        return self.forward_prob(input_x)
    


class PCANetwork(nn.Module):
    """
    Input shape: (N, x)    -- the principal components
    Output shape: (N, 10)  -- class probabilities

    Write this so that the constructor takes in the parameter `x` - 
    the number of features from PCA - and builds layers according to `x`
    """
    def __init__(self, x: int, batch_size = 32):
        pass
    

def train_network(
    net: NormalNetwork | PCANetwork,
    train_data: torch.Tensor, 
    *,
    epochs: int = 20,
    learning_rate: float = 0.01,
) -> None:
    """ 
    Train the neural network using `train_data`.
    `train_data` will be in the shape that the neural network requires. 
    For example, if training NormalNetwork, `train_data` will be in the 
    shape (N, 728). Batching is up to you, and the train_data is separated into iterable batches.
    """
    optimizer = torch.optim.SGD(net.parameters(), lr= learning_rate)

    for epoch in range(epochs):
        for idx, (images, labels) in enumerate(train_data):
            # initialize gradients to zero
            optimizer.zero_grad()
            # Forward pass - inexplicit call to NormalNetwork.forward()
            outputLayer = net(images)
            outputLayer = outputLayer.view(-1, 10)
           
            loss = net.lossFunc(outputLayer, labels)
            # backward propagation
            loss.backward()
            # adjust weights and biases
            optimizer.step()
    
    # save weights and biases for testing
    torch.save(net.state_dict(), "normalNet_weights.pth")


def test_network(
    net: NormalNetwork | PCANetwork,
    test_data: torch.Tensor, 
) -> dict[str, float]:
    """ 
    Test the neural network using `test_data`
    `test_data` follows the same shape as `train_data` as above.
    Return a dictionary of strings with measurements of network 
    performance, like accuracy, f1 score, precision, etc.
    """
    # load weights and biases from training
    net.load_state_dict(torch.load("normalNet_weights.pth"))
    
    # initialize variables
    total_correct = 0
    total_loss = 0
    test_data_size = len(net.test_loader.dataset)
    model_predicted = []
    actual_labels= []
    # iterate through test data without backpropagation and gradient descent
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_data):
            # Forward pass
            outputLayer = net(images)
           
            # sum loss over each batch
            total_loss += net.lossFunc(outputLayer, labels).item()
            
            # if index with max prob is the same index where actual label is 1, then increment total_correct
            predictedLabels = torch.argmax(outputLayer, dim=1)
            labelsMatrix = torch.zeros(labels.shape[0], 10)
            for im_idx in range(labels.shape[0]):
                labelsMatrix[im_idx][labels[im_idx]] = 1
            actualLabels = torch.argmax(labelsMatrix, dim=1)

            total_correct += (predictedLabels == actualLabels).sum().item()
            model_predicted.extend(predictedLabels.tolist())
            actual_labels.extend(actualLabels.tolist())
            # print(f"Correct predictions: {(predictedLabels == actualLabels).sum().item()}/{len(images)}")
    return {
        "accuracy": total_correct / test_data_size, # number of correct predictions / num samples in test dataset
        "loss": total_loss / len(test_data), # summed loss / number of batches
        "confusion matrix": confusion_matrix(actual_labels, model_predicted) 
    }
