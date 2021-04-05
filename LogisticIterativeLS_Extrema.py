#6106 - Statistical Computing 
#Exam 2
#Patrick Rachal

#Problem 2: Iterative Least Squares Algorithm for Logistic Regression
#In the second problem, we explore using Newton’s Method to find the maximum likelihood estimates of the parameter coefficients of a simple logistic regression model. We seek to maximize the likelihood function.

#We begin the Python program by importing numpy for its function, matplotlib for its plotting, pandas to read in the data, and the linear algebra module of numpy for its matrix inverse function.
import numpy as np
import matplotlib as plt
import pandas as pd
from numpy import linalg as LA
#We then read in the challenger dataset. We will use the temp variable to predict the event of an O-ring failure (the failure variable which indicated the number of O-ring failures which occurred on that mission). Due to the file being .txt, we must manually name the columns. Unneeded columns are given filler titles, and the data is stored in an array.
challenger = pd.read_csv('challenger.txt', sep=" ", header=None)
challenger.columns = ["a", "failure", "temp","b","c","d","e"
challenger = np.array(challenger)
#From the array we extract the variables and recode then such at any number of failures simply means the event of failure occurred. For calculation purposes, y is given a new axis and transposed into a vector.
y = np.array(challenger[:,1],copy=True)
for i in range(len(y)):
    if y[i] > 0:
        y[i] = 1
y = y[np.newaxis].T
#We then extract the z predictor variables and concatenate the additional row of ones required for the calculation. 
z = np.array(challenger[:,2],copy=True)[np.newaxis]
ones = np.ones(len(z[0,:]))[np.newaxis]
z = np.concatenate((ones,z),axis=0).T
#We then define the pi_prob function which calculates the probability of an event via the logistic function. It requires the beta coefficients and the predictor variable z.
def pi_prob(z,b):
    a = np.exp( b[0,0] + b[1,0]*z )
    return(a/(1+a))
#We then define the beta_est function, which does most of the work in calculating the estimated beta coefficients. It requires the  vector, the  vector, and the p precision scalar which specifies the convergence condition.
def beta_est(z,y,p):
#The function first determines how many observations there are in the dataset and initializes the convergence criterion C at  so the while loop can begin.
    n = len(y)
    C = (1,1)
    beta = np.array([.001,.001],dtype=float)[np.newaxis].T
    betas = [(.001,.001)]
    while LA.norm(C) > p:   
        w_t = np.identity(n)
        prob_t = np.zeros(n,dtype=float)[np.newaxis].T
        for j in range(n):
            prob_t[j][0] = pi_prob(z[j][1],beta)
            w_t[j][j] = prob_t[j][0]*(1-prob_t[j][0])
        A = LA.inv(np.matmul(np.matmul(z.T,w_t),z))            
        B = np.matmul(z.T,(y-prob_t))     
        C = np.matmul(A,B)
        beta = beta + C
        betas.append((float(beta[0]),float(beta[1])))
#The values are then passed back to the beginning of the loop until R loops have been completed, when the function returns the final estimates of the betas, as well a matrix containing the iterative results.
    return(betas,beta)
#We call the function, storing the final values as well as the iterative results which we plot.
beta_hats,beta_hat = beta_est(z,y,10)
plt.pyplot.plot(beta_hats)
plt.pyplot.xlabel("Index")
plt.pyplot.ylabel("Values")
plt.pyplot.title("Beta Estimates")
 
#It seems that estimates only require around 5 iterations to level out.
#We finally apply the logistic pi_prob function to determine that O-ring failure is almost certain given a temperature of 31, as the probability of failure given 31 is very close to one.
pi_prob(31,beta_hat)
#Out[49]: 0.9999959110825846


#Problem 3: Finding Extrema
#In this third problem we explore the different methods of finding minimum and maximum values of function using different methods. We explore using Newton’s Method for functions of one and two variables, as well as the steepest descent method and secant method.
#The method of steepest descent involves using gradient, which indicates the direction of steepest descent given a value. Once that direction is determined, one can use the Golden Section Search to find the minimum value along that direction, i.e., a univariate minimization problem. 
The secant method involves replacing the Hessian matrix in Newton’s Method with an estimate.


#Problem 3, Part 1: Minimizing A Multivariate Function
#Problem 3, Part 1.a: Newton’s Method for Multivariate Functions
#We start the program by importing numpy and its linear algebra module for their functions, and matplotlib for its plotting to check our work.

import numpy as np
import matplotlib as plt
from numpy import linalg as LA
#We then define the function g, as well as the gradient and Hessian.
def g(x,y):
    return (4*x*y + (x + y**2)**2)
def g_prime(x,y):
    return(np.matrix((4*y + 2*(x + y**2),4*x + 4*y*(x + y**2) )  ).T )
def g_dub_prime(x,y):
    top = np.matrix(([2],[4 + 4*y])).T
    bottom = np.matrix(([4 + 4*y],[12*(y**2)])).T
    return(np.concatenate((top,bottom),axis=0))
#We then define the function Newton, which takes in the starting “guess” values x and y.
def Newton(x,y):
    d = 1
    step = np.array((x,y),copy=True)[np.newaxis].T
    steps = [(x,y)]
    while LA.norm(d) > .0001:
        d = np.matmul(LA.inv(g_dub_prime(x,y)),g_prime(x,y))
        step = step - d
        x = float(step[0][0])
        y = float(step[1][0])
        steps.append((x,y))
    return(steps,step)

N_steps = Newton(3,4)[0]
N_step = Newton(3,4)[1]
N_steps = np.array(N_steps)
plt.pyplot.plot(N_steps)
plt.pyplot.xlabel("Index")
plt.pyplot.ylabel("Values")
plt.pyplot.title("Convergence")
plt.pyplot.legend(('X', 'Y'))
plt.pyplot.show()
print("Convergence Occurred at",N_step[0],N_step[1])
 
#Convergence Occurred at [[0.88886704]] [[-0.66663388]]

#Problem 3, Part 1.b: Method of Steepest Descent Via The Golden Section
#In part b, we attempt the same problem using a different method. As mentioned above, the gradient vector points in the direction of steepest descent. We can use that direction to plot a straight line through the multivariate surface and find the local minimum along that line. Once the minimum is found, we calculate a new gradient and a new minimum along that direction. 

#The first defined function step_func takes in a point on the xy plain and handles most of the work as detailed below.
def step_func(x,y):
#It first defines :
    gr = (np.sqrt(5)+1)/2
#And then defines the area to be searched. Since the parameterization sets , we search the neighborhood around .
    a,b = x-1,x+1
#It then determines the line of interest using the gradient.
    m = float(g_prime(x,y)[1])/ float(g_prime(x,y)[0])
    s = y - m*x
    def f(t):
        return (g(t,m*t + s))
#The function then performs the line search via the Golden Section as described above.
    while ( np.abs((f(b) - f(a))) > .0003 ):
        t2 = a + (b-a)/gr
        t1 = b - (b-a)/gr
        if ( f(t1) < f(t2) ):
            b = t2
        else:
            a = t1
    hold = (a+b)/2
#The point determined is a point along the x-axis. The function returns this point along with the corresponding y-value on the line of interest. This is the minimum point along the line.
    return(hold, m*hold + s)
#The Steep function takes in the first  guess, and then loops over step_func, storing the iterative values and checking for convergence.
def Steep(x,y):
#We start by storing the point  as the first step, and then the guess as the second. This is so the while loop has something to check upon first entry.
    steps = [(0,0)]
    steps.append((x,y))
#The while loop checks for convergence by seeing if the norm of the two most recent iterations have a difference of more than . If the difference is smaller, we consider the values converged.
    while np.abs(LA.norm(steps[-1]) - LA.norm(steps[-2])) > .00003:
   #are passed to step_func, who updates their values. The new values are appended to the steps list, and the loop starts over.
        x,y = step_func(x,y)
        steps.append((x,y))
#Once convergence has been obtained, we store the results in final. The steps and final are returned by the function.
    final = np.array((x,y),copy=True)
    return(steps,final)
#We call the function with the same first guess and plot the results, trimming off the irrelevant  step.
Steep_steps = Steep(3,4)
values = np.array(Steep_steps[0])[range(1,len(Steep_steps[0])),:]
plt.pyplot.plot(values)
plt.pyplot.xlabel("Index")
plt.pyplot.ylabel("Values")
plt.pyplot.title("Convergence")
plt.pyplot.legend(('X', 'Y'))
plt.pyplot.show()
print("Convergence Occurred at",Steep_steps[1])
 
#Convergence Occurred at [ 0.88910389 -0.66722447]
#We also plot a scatter plot:
xs, ys, = values[:,0],values[:,1]
plt.pyplot.scatter(xs,ys)
plt.pyplot.xlabel("X Values")
plt.pyplot.ylabel("Y Values")
plt.pyplot.title("Convergence")
plt.pyplot.show()
 
#Problem 3, Part 2.a: Newton for a Univariate Function
#We start the Python program by defining  and its derivatives.
def h(x):
    return(np.log(x)/(1+x))
def h_prime(x):
    return (1/x - np.log(x)/(1+x))
def h_dub_prime(x):
    a = 2*np.log(x)/(1+x)
    b = (3*x + 1)/(x**2)
    return (a-b)
#We then define Newton2, which takes in a single point .
def Newton2(x):
#The convergence criteria epsilon is initialized at 1, the first step is set to the input, and stored in the step list.
    epsilon = 1
    steps = [x]
#The while loop runs until the difference between the steps is less than .
    while epsilon > .0001:
#The difference is calculated as indicated, the first derivative at the most recent step divided by the second derivative at that step.
        d = h_prime(steps[-1])/h_dub_prime(steps[-1])
#The absolute value of the difference is the convergence indicator epsilon.
        epsilon = np.abs(d)
#The next step is the previous step minus the difference. The next step is added to the list of steps, and the loop returns to the beginning.
        steps.append(steps[-1] - d)
#Once convergence has occurred, we return the list of steps.        
    return(steps)
#We call the function and plot the results.
N_steps2 = Newton2(5)
plt.pyplot.scatter(range(len(N_steps2)),N_steps2)
plt.pyplot.xlabel("Index")
plt.pyplot.ylabel("Value")
plt.pyplot.show()
print("Convergence Occurred at",N_steps2[-1])
 
#Convergence Occurred at 3.591444489511332
#Problem 3, Part 2.b: The Secant Method
#The secant method proceeds identically to Newton’s Method, but an estimation of the second derivative is used, making it useful for situations when the second derivative is difficult to calculate.

#Note this means we use the last two steps to estimate the next step. Hence, the function secant takes in to initial guesses.
def secant(x0,x1):
    epsilon = 1
#After epsilon is initialized, the first two guesses are stored in the steps list.
    steps = [x0,x1]
#The while loop continues until the difference is less than .
    while epsilon > .0001:
#Due to the complexity of the new difference, it is calculated in pieces. First is the estimate of the second derivative, stored in d0.
        d0 = ( h_prime(steps[-1]) - h_prime(steps[-2]) ) / (steps[-1] - steps[-2])
#Then the total difference is calculated.
        d = h_prime(steps[-1]) / d0 
#Epsilon is calculated to be checked by the while loop.
        epsilon = np.abs(d)
#The next step is added to the list of steps, and the loop returns to the beginning.
        steps.append(steps[-1] - d)
#The function then returns the list of steps.
    return(steps)
#We then call the function with the first guess  and plot the results.
sec_steps = secant(1,2)
plt.pyplot.scatter(range(len(sec_steps)),sec_steps)
plt.pyplot.xlabel("Index")
plt.pyplot.ylabel("Value")
plt.pyplot.show()
print("Convergence Occurred at",sec_steps[-1])
 
#Convergence Occurred at 3.5911214766527335
