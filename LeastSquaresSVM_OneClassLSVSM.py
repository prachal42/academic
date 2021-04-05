#Statistical Computing
#Project 1
#Patrick Rachal
#Problem 1:
#	In the first program, we approach a classification problem by means of a Least Squares Support Vector Machine. 

#Our first step is to import the necessary packages. We’ll need numpy for various math functions (specifically linalg for its linear algebra) and pandas for its csv import capabilities.
import numpy as np
from numpy import linalg as LA
import pandas as pd
#We then import our datafile, and code our qualitative response:
charlie_data = pd.read_csv("charlie.csv")
charlie_data['y'] = np.where(charlie_data['Data'] == 'Original', 1, -1)
#We define the function est(), which will return the proportion of correctly classified observations.
def est():
#This problem has two parts: one in which we solve the SVM, and the second in which we use training/testing data to validate our work. This program contains a Boolean variable train, which when True randomly separates the data into charlie_test containing 7 observations and charlie_train containing 23 observations. When train is set to False, it defines charlie_test and charlie_train as the complete dataset. That is, the program doesn’t distinguish between the two. The variable check is a vector of randomly determined True/False values that allows a random sample of the correct size every run.
if train == True:
    check = np.random.permutation(30) > 6
    charlie_train = charlie_data[check]
    charlie_test = charlie_data[~check]    
else:
    charlie_train = charlie_data
    charlie_test = charlie_data
#Part I: Solving the LS SVM
#For our first goal, which is simply solving the SVM, we set train to False. We force our complete dataset into an array named char, and determine its length n.
char = np.asarray(charlie_train)
n = len(char)
#We then gather all the s into a vector named y which is forced into the float type, and similarly make yt, the transposed vector of s. We also separate our predictor variables into the matrix x.
y = char[:,8]
y = y.astype(np.float)
yt = y[np.newaxis]
yt = np.transpose(yt)
x = char[:,(2,3,4,5)]

#We also go ahead and separate the s and s for testing (since train is False, these represent all and values from the complete data set):
test_x = np.array(charlie_test)[:,(2,3,4,5)]
test_y = np.array(charlie_test)[:,8]
#We then select our and , and define the Kernel function:
c = 2
sig = 1
def Kern(x1,x2):
    return(np.exp(-(LA.norm(x1-x2)**2)/sig))
#To build the omega matrix, we first determine its size and initialize it full of zeros. We then use nested for loops to populate it according to the algorithm. 
omg = [[0 for x in range(n)] for y in range(n)]
for i in range(n):
    for j in range(n):
        omg[i][j] = y[i] * y[j] * Kern(x[i], x[j])
#We then add
H = (omg+((1/c)*np.identity(n)))
#To build the large matrix piecewise, we first construct the single top row, and then the bottom. For the top row, we define a matrix consisting of a single zero and then concatenate it to the array containing observed s.
a = np.array([0])
top = np.concatenate([a,y])[np.newaxis]
#For the bottom, we concatenate the H matrix to the transposed s.
bottom = np.concatenate((yt,H),axis=1)
#We then concatenate top to bottom, to form complete.
complete = np.concatenate((top,bottom),axis=0)
#This matrix is then inverted.
inverse = LA.inv(complete)

#We then create the matrix by initializing a matrix full of zeros, then populating all but the first indices with 1. The matrix is then transposed for later multiplication.
z = np.zeros(n+1)
for i in range(1, n+1):
    z[i] = 1
z = np.transpose(z[np.newaxis])
#We can now solve:
balphas = np.matmul(inverse,z)
#We can now apply the classification function to all the points in our dataset, and store them in the array yi. (Reminder: since train = False, test_x contains all prediction observations from the dataset. 
yi = [0 for i in range(len(test_x))]
for i in range(len(test_x)):
    yi[i] = balphas[0]
    for j in range(n):
        yi[i] = yi[i]+(y[j]*balphas[j+1]*Kern(x[j],test_x[i]))
#As expected, all data points are classified correctly.
return(np.mean(np.transpose(np.sign(yi))==test_y))
print(est())

#Part II: Training and Predicting
#The second part uses mostly the same code. We start by turning the train variable to True. The data is randomly separated into charlie_test containing 7 observations and charlie_train containing 23 observations via the random partition function.
if train == True:
    check = np.random.permutation(30) > 6
    charlie_train = charlie_data[check]
    charlie_test = charlie_data[~check]
#This has an impact on the program flow. The matrix char is only made up of the 23 training observations:
char = np.asarray(charlie_train)
n = len(char)
#The y, yt, and x matrices therefore also only contain training data. This is very important.
y = char[:,8]
y = y.astype(np.float)
yt = y[np.newaxis]
yt = np.transpose(yt)
x = char[:,(2,3,4,5)]
#The test_x and test_y arrays contain the s and s not in the training data, as they will be used to test our support vector machine.
test_x = np.array(charlie_test)[:,(2,3,4,5)]
test_y = np.array(charlie_test)[:,8]
#The parameters  and  are chosen just as before, and the Kernel function definition does not change. The choice of  and  came about by trial and error to maximize the correct classification rate.
c = 10
sig = 100
def Kern(x1,x2):
    return(np.exp(-(LA.norm(x1-x2)**2)/sig))
#The omega and H matrices is defined the same way, but are now only made with observations from the training data.
omg = [[0 for x in range(n)] for y in range(n)]
for i in range(n):
    for j in range(n):
        omg[i][j] = y[i] * y[j] * K(x[i], x[j])
H = (omg+((1/c)*np.identity(n)))

#The same strategy as Part I applies in solving for 
a = np.array([0])
top = np.concatenate([a,y])[np.newaxis]
bottom = np.concatenate((yt,H),axis=1)
complete = np.concatenate((top,bottom),axis=0)
inverse = LA.inv(complete)
z = np.zeros(n+1)
for i in range(1, n+1):
    z[i] = 1
z = np.transpose(z[np.newaxis])
balphas = np.matmul(inverse,z)
#The array of b and the alphas is finished, and we can apply the classification function with these estimates, to the test_xs which were not used in their construction:
yi = [0 for i in range(len(test_x))]
for i in range(len(test_x)):
    yi[i] = balphas[0]
    for j in range(n):
        yi[i] = yi[i]+(y[j]*balphas[j+1]*K(x[j],test_x[i]))
return(np.mean(np.transpose(np.sign(yi))==test_y))
#To determine the efficiency of our simulation, we define an empty list, and populate it with 1000 runs of the function and determine the mean:
test = []
for i in range(1000):
    test.insert(i,est())
print(np.mean(test))
#0.8994285714285714
#So our classifier averages 89.9% classification accuracy.

#Problem 2:
#For the second problem, our goal is to use the same data set to build a One-Class Least-Squares Support Vector Machine. 

#Our Python program begins with the standard imports: numpy for its math functions, (specifically linalg for matrix multiplication and inversing), pandas for importing the csv, and matplotlib for its plotting functions.
import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

#We pull in the Charlie data, code the  responses, and partition the data into “Original” and “New” data sets. We really only need the “Original” data, but to test our program we’ll later look at the “New” data.
charlie_data = pd.read_csv("charlie.csv")
charlie_data['y'] = np.where(charlie_data['Data'] == 'Original', 1, -1)
charlie_all = np.array(charlie_data)
charlie_new = charlie_data.loc[charlie_data['y'] == -1]
charlie_org = charlie_data.loc[charlie_data['y'] == 1]
#First, to get a visualization we plot a scatterplot of the data. Yellow dots are “Original”, purple dots are “New”. Remember we only train with “Original” data points.
plt.scatter(charlie_all[:,6],charlie_all[:,7],c=charlie_all[:,8])
 
#For future validation, the s and s are put into arrays:
new_z = np.array(charlie_new)[:,(6,7)]
new_y = np.array(charlie_new)[:,8]
#The Kernel function is defined (same as in problem 1), and  and  are assigned values globally so they can be changed easily.

c = 2
sig = 1
def Kern(x1,x2):
    return(np.exp(-(LA.norm(x1-x2)**2)/sig))
#The original data is turned into a numpy array and assigned the name char. Its length is saved as n, as it is referenced frequently throughout the program.
char = np.asarray(charlie_org)
n = len(char)
#From char, we pull the  array and the array containing both  and .
z = char[:,(6,7)]
zy =char[:,(6,7,8)]
#The K matrix is defined full of zeros, and populated by nested for loops
K = [[0 for x in range(n)] for y in range(n)]
for i in range(n):
    for j in range(n):
        K[i][j] = Kern(z[i], z[j])
#The H matrix is then created by adding the -reciprocal identity matrix.
H = (K+((1/c)*np.identity(n)))
 is inverted, e is defined, and then and the alphas are calculated as stated above.
Hi = LA.inv(H)
e = np.transpose(np.full((1,n),1))
roe = -1 / ( np.matmul(np.matmul(np.transpose(e),Hi),e))
alphas = -roe*np.matmul(Hi,e)

#We now have all the elements for the classification function, but in order to normalize testing data we need to calculate  and . The array yi is defined and populated according to the classification function.
yi = [0 for i in range(n)]
for i in range(n):
    yi[i] = roe
    for j in range(n):
        yi[i] = yi[i] + (alphas[j] * Kern(z[j], z[i]) )
#We then find the maximum and minimum values using numpy.
yi_min = np.amin(yi)
yi_max = np.amax(yi)
#In order to test given data, we define the classification function in Python as y(t). This returns the non-normalized classification of the input t.
def y(t):
    x = roe
    for i in range(n):
        x = x + (alphas[i] * Kern(z[i],t))
    return(x)
#We then define the normalizing function y_bar(t), which relies on the function y(t), and the maximum and minimum values found earlier.
def y_bar(t):
    return(np.sign(1-((2*np.abs(y(t))) / (c*((np.sign(y(t))*(yi_max+yi_min))+(yi_max-yi_min))))))
#We now have all we need to test the “New” data and see how our OCLSSVM performs. We define the array new_z_bar, which we will populate with the normalized classification of the “New” Z data using a for loop and the previously defined functions.
new_z_bar = np.array([0 for x in range(len(new_z))])
for i in range(len(new_z)):
    new_z_bar[i] = y_bar(new_z[i])

#We then compare the true  values to the predicted values, and determine our classification rate.
print(new_z_bar)
print(new_y)
print(np.mean(new_z_bar == new_y))
#The correct classification rate using our OCLSSVM is 0.6. Given that four of the ten “New” observations are intermingled with the “Original” observations (See figure 1), this is to be expected. By adjusting c and sigma, we could increase our correct classification rate, but we would also lose the generalization of the model. That is, we would overfit.
#We now plot the boundary determined by the OCLSSVM. We will do this by classifying 800,000 points surrounding the training data. We start by defining the arrays containing the points of interest.
x1 = np.array(np.arange(-4,6,.01))
x2 = np.array(np.arange(-4,4,.01))
#We then define the array p, which will contain the points of interest, as well as the predicted outcome. We will also force is to be of the type float.
p = [[0 for x in range(3)] for y in range(len(x2)*len(x1))]
p = np.array(p, dtype=float)
#The array p is populated using nested for loops.
for j in range(len(x2)):
    for i in range(len(x1)):
        p[(len(x1)*j)+i][0] = x1[i]
        p[(len(x1)*j)+i][1] = x2[j]
        p[(len(x1)*j)+i][2] = y_bar((x1[i],x2[j]))
#In order to view these calculated points simultaneously with the training data, we will concatenate the “Original” s and s to p. However, we will change the “classification” of the original s to 10 to highlight their presence on the graph.
for i in range(len(z)):
    zy[i][2] = 10
p = np.concatenate((p,zy), axis=0)

#We then use matplotlib to print a scatterplot.
plt.scatter(p[:,0],p[:,1],c=p[:,2])
plt.xlabel("z_1")
plt.ylabel("z_2", rotation=0)
plt.title(“c=%.2f, sigma=%.2f” % (c, sig))
#With the following results shown in Figure 2:
 
#Figure 2
#The yellow dots represent the “Original” data points used for building the OCLSSVM, the light purple area represents values not considered outliers, and the dark purple area represents values which are. 