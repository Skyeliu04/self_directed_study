import numpy as np

def foo_gradient(x, y):
    """
    Computes the gradient of foo(x,y) = sin(cos(x)+sin(2y))
    Returns (dfdx, dfdy) as computed in Task 3
    """
    dfdx = -np.cos(np.cos(x) + np.sin(2*y)) * np.sin(x)
    dfdy = 2 * np.cos(np.cos(x) + np.sin(2*y)) * np.cos(2*y)
    
    return (dfdx, dfdy)

def gradient_descent(function, gradient, x1, y1, eta, epsilon):
    """
    Implements gradient descent algorithm
    Args:
        function: the function to minimize
        gradient: function that returns the gradient
        x1, y1: starting point
        eta: learning rate
        epsilon: convergence threshold
    Returns:
        (x_min, y_min, history): minimum point found and history of points visited
    """
    history = [(x1, y1)]
    x_current, y_current = x1, y1
    
    while True:
        # Get gradient at current point
        (dx, dy) = gradient(x_current, y_current)
        
        # Calculate new point
        x_new = x_current - eta * dx
        y_new = y_current - eta * dy
        
        # Add new point to history
        history.append((x_new, y_new))
        
        # Check convergence
        if abs(function(x_new, y_new) - function(x_current, y_current)) < epsilon:
            return (x_new, y_new, history)
        
        # Update current point
        x_current, y_current = x_new, y_new