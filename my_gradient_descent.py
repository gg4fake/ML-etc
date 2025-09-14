import numpy as np
import inspect
from functools import wraps


def gradient(f,x,epsilon = 0.001):
    """
    Calculates gradient of f at x
    ------------------------

    f : function of choice
    x : where the computation takes place
    epsilon : small difference
    -------------------------

    Uses the two point method
    """

    try:
        dim = len(x)

        Gradient = np.zeros(dim)

        for i in range(dim):
            y = x.copy()
            z = x.copy()
            y[i]+=epsilon
            z[i]-=epsilon
            partial = (f(y)-f(z))/(2*epsilon)

            Gradient[i] = partial

        return Gradient

    except: 

        print("Something went terribly wrong in the gradient calculation")


def vectorize_args(func): # For vectorizing f_vec
            """Wrap func(x1, x2, ..., xn) -> f([x1, ..., xn])."""
            @wraps(func)
            def wrapper(vec):
                return func(*vec)
            return wrapper

def gradient_descent_random(f,ite,zeta,dr = 0.99,stopping = 0.01, low =-1, high = 1,epsilon_gradient = 0.0001,max = 20,Full =True, decay = True):
    """
    gradient descent with random starting points and a decay rate decay rate
    ------------------------
    f : function of choice (does not have to be vectorized)
    ite : iterations
    zeta: scale
    dr : decay rate
    low & high : the max and min which upon the guesses are drawn upon for every param.
    epsilon_gradient : small difference passed to the gradient
    Full : return epoch or no
    decay : vestigial 
    -----------------------
    """
    try:
        sig = inspect.signature(f)
        params = sig.parameters
        dim = len(params)
        f_vec = vectorize_args(f)
        runs = []
        starts = np.random.uniform(low, high,size=(ite,dim))

        steps =[]
    
        for x in starts:
            t =zeta
            v = x.copy()
            step =[]
            for i in range(max):
            
                q=v - t*gradient(f_vec,v)
                if np.linalg.norm(q-v)< stopping:
                    break
                else:
                    v = q 
            
                t = t*dr 
                step.append(v)
            steps.append(step)
            runs.append(v)

        runs = np.array(runs)
        outcomes = [f_vec(i) for i in runs]

        if not Full: 

            return min(outcomes)
        else: 
            ind =np.argsort(outcomes)
            return min(outcomes),runs[ind], outcomes,steps
    
    except:
        print("there was an error in the gradient descent")
