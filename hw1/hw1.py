import numpy as np
import itertools
np.random.seed(42)

def preprocess(X, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column. 

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The scaled labels.
    """
    X_mean = X.mean(axis = 0)
    X_max = X.max(axis = 0)
    X_min = X.min(axis = 0)
    
    X = (X - X_mean) / (X_max - X_min)
    y = (y - np.mean(y)) / (np.max(y) - np.min(y))
    
    return X, y

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.  

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    hFunction = np.matmul(X, theta)
    squareError = np.square(hFunction - y) # calculate the the squared errors
    J = np.sum(squareError) / (2 * len(y)) # calculate J according to the formula
    
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by 
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = []
    scalar = alpha / len(y)
    
    for i in range(num_iters):
        
        hFunction = np.matmul(X, theta)
        theta = theta - ((np.matmul((hFunction - y), X)) * scalar)
        J_history.append(compute_cost(X, y, theta)) # calculate and add the loss value
    
    return theta, J_history


def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """
    matrixInverse = np.linalg.inv(np.matmul(X.transpose(), X))
    pinv_X = np.matmul(matrixInverse, X.transpose())       
    pinv_theta = np.matmul(pinv_X, y)
    
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value is smaller than 1e-8. This function is
    very similar to the gradient descent function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = [] # Use a python list to save cost in every iteration
    
    epsilon = 1e-8
    scalar = alpha / len(y)
    
    for i in range(num_iters):
         
        hFunction = np.matmul(X, theta) # multiply theta and X
        theta = theta - (np.matmul((hFunction - y), X) * scalar) # changing thetah according to the formula               
        J_history.append(compute_cost(X, y, theta)) # calculate and add the loss value
        
        if ( i > 0 ) and ( epsilon > (J_history[i-1] - J_history[i]) ) :
            #print("number of iterations = " + str(i))
            return theta, J_history 
    
    return theta, J_history

def find_best_alpha(X, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary 
    with alpha as the key and the final loss as the value.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training 
    for every value of alpha.
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    theta = np.random.random(size=2) # generate start values for theta
    
    for i in range(len(alphas)):
        
        temp, current_loss = efficient_gradient_descent(X, y, theta, alphas[i], iterations)
        alpha_dict.update({alphas[i] : current_loss[-1]})
    
    return alpha_dict

def generate_triplets(X):
    """
    generate all possible sets of three features out of all relevant features
    available from the given dataset X. You might want to use the itertools
    python library.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """
    triplets = []
    triplets = list(itertools.combinations(X, 3))
    
    return triplets

def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best 
    minimizes the cost function. For better performance, you should use the 
    efficient implementation of gradient descent. You should first preprocess
    the data and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A dataframe that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    
    best_triplet = []
    min_cost = 10000
    y = np.array(df['price'])
    theta = np.random.random(size=4)
    
    for element in triplets:
        columns_to_use = [element[0], element[1], element[2]]
        X = np.array(df[columns_to_use])
        y = np.array(df['price'])
        
        X, y = preprocess(X, y)
        X = np.insert(X, 0, 1., axis = 1) # bias trick
        
        _, J_history = efficient_gradient_descent(X, y, theta, alpha, num_iter)
        current_cost = J_history[-1]
        
        if current_cost < min_cost:
            min_cost = current_cost
            best_triplet = list(element)
    
    
    return best_triplet
