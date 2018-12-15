import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  data_loss = 0.0
  #initialize probabilities with zeros
  probabilities = np.zeros((num_train,num_classes))

  for i in range(num_train):
    scores = np.dot(X[i,:],W)
    const = np.max(scores)
    scores = scores - const
    probabilities[i,:] = np.exp(scores)/np.sum(np.exp(scores))
    log_prob = -np.log(probabilities[i,:])

    correct_probs = log_prob[y[i]]
    probabilities[i,y[i]] -= 1.00
    data_loss += correct_probs
        
  data_loss /= num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  dW = (np.dot(X.T,probabilities))/num_train
  dW += reg*W #dReg = reg*W




  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  scores = X.dot(W)
  const = np.max(scores,axis=1,keepdims = True)

  scores = scores - const
  probabilities = np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)

  epsilon = 1e-14
  N = X.shape[0]
  loss = -np.sum(np.log(probabilities[np.arange(N), y] + epsilon)) / N

  dscores = probabilities.copy()
  dscores[np.arange(N), y] -= 1
  dscores /= N
  dW = np.dot(X.T, dscores)


  return loss, dW

