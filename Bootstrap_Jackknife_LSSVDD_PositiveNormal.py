#Statistical Computing Exam I
#Patrick Rachal
#Problem 1, Part I: Bootstrap
#In this first problem, we explore Bootstrap and Jackknife resampling methods. Using the mt_cars data, we use Bootstrap and Jackknife to find estimates of the coefficient of variation of the mpg variable and determine their bias and standard error.

#For the Python program, we start by importing the continuously useful numpy library as well as the pandas library so we can import the dataset csv. 
import numpy as np
import pandas as pd
mtcars_data = pd.read_csv("mtcars.csv")
#We then separate the variable we are interested in.
mpgs = np.transpose(np.array(mtcars_data)[:,1])
coff_o_var_hat = np.std(mpgs,ddof=1)/np.mean(mpgs)
#We then determine the number of bootstrap samples we’ll use, B.

B = 50
#We then define the function Bootstrap, which takes in the size parameter R and returns the specified number of bootstrap estimates of the coefficient of variation. It declares an array full of zeros called BS which it will populate with the estimates. It then goes into a for loop for each estimate. Within the loop, it generates a random array of indices which it uses to sample with replacement from the original sample mpgs. It defines the array samp, which is populated with the bootstrap sample inside another for loop. It then determines the coefficient of variation estimate for that bootstrap sample and assigns it to the BS array. Once BS is filled with R bootstrap estimates, the function returns the array.

def Bootstrap(R):
    BS = [0 for i in range(R)]
    for i in range(R):
        rand = np.random.randint(0,len(mpgs),len(mpgs))
        samp = [0 for i in range(len(mpgs))]
        for j in range(len(mpgs)):
            samp[j] = mpgs[rand[j]]
        BS[i] = (np.std(samp,ddof=1)/np.mean(samp))
    return(BS) 
#Using this function, we declare the array z_samp and fill it with . We then determine .
z_samp = Bootstrap(B)
z_star_hat_bar = np.mean(z_samp)
#We then calculate and print the bootstrap bias and standard error.
BS_Bias = z_star_hat_bar - coff_o_var_hat
BS_Err = np.sqrt((1/(B-1))*sum((z_samp - z_star_hat_bar)**2))
print("\nThe Bootstrap Estimated Bias is",BS_Bias,"\n")
print("The Bootstrap Estimated Standard Error is",BS_Err,"\n")


#The following output is printed:
#The Bootstrap Estimated Bias is -0.006727931670658549 
#The Bootstrap Estimated Standard Error is 0.03058063612901005

#Problem 1, Part II: Jackknife
#We proceed to the Jackknife method in Part II. Jackknife is another resampling method which relies on a “leave-out-one” methodology similar to Cross-Validation. Estimates for  are made by considering the sample with one observation removed, and this is done for every  observations. 

#In Python, we begin by declaring JN, an array which we will populate with the n jackknife estimates.
JN = [0 for i in range(len(mpgs))]
#We then use a for loop to populate the array, taking advantage of numpy’s omit function.
for i in range(len(mpgs)):
    omit = np.delete(mpgs,i)
    JN[i] = (np.std(omit,ddof=1)/np.mean(omit))

theta_hat_dot_bar = np.mean(JN)
#We then determine and print the bias and standard error of the estimates.
JN_Bias = ((len(mpgs)-1)*((theta_hat_dot_bar - coff_o_var_hat)))
JN_Err = np.sqrt(((len(mpgs)-1)/len(mpgs))*sum((JN - theta_hat_dot_bar)**2))
print("The Jackknife Estimated Bias is",JN_Bias,"\n")
print("The Jackknife Estimated Standard Error is",JN_Err,"\n")
#With the following output:
#The Jackknife Estimated Bias is -0.0025778903923485696 
#The Jackknife Estimated Standard Error is 0.033633357489055535

#Problem 1, Part III: Bootstrap Confidence Intervals
#We derive confidence intervals for the bootstrap estimate of the coefficient of variation using the t-distribution. We determine bootstrap t-statistics.

#In Python, we declare a function called Bootstrap_t, which takes in the size parameter R. The function will return the bootstrap sample estimates, as well as the standard error for each individual sample. The function begins by defining empty arrays with which to fill with these estimates and standard errors. 
def Bootstrap_t(R):
    BS = [0 for i in range(R)]
    se_b = [0 for i in range(R)]
#The function then begins its first for loop, which generates the random indices for the bootstrap sample, and declares an array full of zeros to fill with the bootstrap sample. In the second loop, the bootstrap sample is generated. The estimate for the coefficient of variation is then stored in the BS array.
    for i in range(R):
        rand_1 = np.random.randint(0,len(mpgs),len(mpgs))
        samp_1 = [0 for i in range(len(mpgs))]
        for j in range(len(mpgs)):
            samp_1[j] = mpgs[rand_1[j]]
        BS[i] = (np.std(samp_1,ddof=1)/np.mean(samp_1))
#The function then resamples from the bootstrap sample in the same method and stores the bootstrapped bootstrap in the declared array BS2. 
        BS2 = [0 for i in range(R)]
        for k in range(R):
            rand_2 = np.random.randint(0,len(mpgs),len(mpgs))
            samp_2 = [0 for i in range(len(mpgs))]
            for l in range(len(mpgs)):
                samp_2[l] = samp_1[rand_2[l]]
            BS2[k] = (np.std(samp_2,ddof=1)/np.mean(samp_2))
#Using the values in BS2, we can determine .   
        se_b[i] = np.sqrt((1/(R-1))*sum((BS2 - BS[i])**2))
#Which we then use to determine the s. They are paired together with the bootstrap estimates and returned.
    t_b = (BS - coff_o_var_hat) / se_b
    res = [np.array(BS),t_b]
    return(res)
#We calculate result, which is B of such ,  pairs, and split them into two arrays BS_samp and BS_t. We then determine , the standard deviation of the replicates , and store it in BS_t_Err. 
result = Bootstrap_t(B)
BS_samp = result[0]
BS_t_Err = np.sqrt((1/(B-1))*sum((BS_samp - np.mean(BS_samp))**2))
BS_t = result[1]
#We then sort BS_t, and determine the percentiles  and .
BS_t = np.sort(BS_t)
t_l = np.percentile(BS_t,2.5)
t_u = np.percentile(BS_t,97.5)
#We then calculate and print the interval according to the formula.
print("Bootstrap t-Interval")
print("We are 95% confident that the true Coefficent of Variation is between", coff_o_var_hat-(t_u*BS_t_Err),"and",coff_o_var_hat-(t_l*BS_t_Err),"\n")
#With the following output:
#Bootstrap t-Interval
#We are 95% confident that the true Coefficent of Variation is between 0.27292032913952474 and 0.3615037722721631
#Calculating the Percentile Confidence Interval is much simpler. We recall the Bootstrap function defined in the first part of Problem 1. We sample 1000 Bootstrap estimates and place them in the array Percentile, sort them, and determine the 25th and 975th observations (corresponding to the 2.5th and 97.5th percentile).

Percentile = Bootstrap(1000)
Percentile = np.sort(Percentile)
print("Bootstrap Percentile Interval")
print("We are 95% confident that the true Coefficent of Variation is between",Percentile[25],"and",Percentile[975])
#With the following output:
#Bootstrap Percentile Interval
#We are 95% confident that the true Coefficent of Variation is between 0.22240653587042955 and 0.3563428961919287
#So the Bootstrap t-Interval is slightly smaller than the Bootstrap Percentile Interval.

#Problem 2, Part I: Solving the LS-SVDD
#In the first problem, we approach a classification problem by means of a Least Squares Support Vector Data Descriptor. We In our given data set, we have 20 observations, with 2 predictor variables which we use to predict if a test observation is the in the class “Original”. The model seeks to determine a hypersphere which contains the given training observations. If a test observation is contained by the sphere, we conclude it is “Original”. We will also test the 10 observations from the “New” class to evaluate our model.

#For a test vector z, we calculate its distance from the center of the hypersphere.

#The Python program starts with import numpy for its functions, pandas for to import the file, and matplotlib for the plot at the end to plot the boundary determined. We also import the linear algebra portion of numpy for matrix multiplication.
import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
#We import charlie.csv and call it charlie_data. We then assign the outcome variables 1 and -1 to the Original and New data respectively. We then split the dataset into New and Original portions.
charlie_data = pd.read_csv("charlie.csv")
charlie_data['y'] = np.where(charlie_data['Data'] == 'Original', 1, -1)
charlie_new = charlie_data.loc[charlie_data['y'] == -1]
charlie_org = charlie_data.loc[charlie_data['y'] == 1]
#We force the original data into a numpy array and determine its size as it will be frequently referenced throughout the code.
char = np.asarray(charlie_org)
n = len(char)
#We separate the Z variables associated with the Original outcome into an array called z, and then separate the variables and the outcomes into an array called zy.
z = char[:,(6,7)]
zy = char[:,(6,7,8)] 
#We then similiary separate the variables and outcomes associated with the New outcome.
char_new = np.asarray(charlie_new)
new_z = char_new[:,(6,7)]
new_zy = char_new[:,(6,7,8)]
#We globally declare the c and sigma associated with the Gaussian Kernel so that it will be simple to keep consistent. We then define the Gaussian Kernel function taking advantage of the numpy linear algebra norm function.
c = 2
sig = 5
def Kern(x1,x2):
    return(np.exp(-(LA.norm(x1-x2)**2)/sig))
#We use nested for loops to populate the K matrix.
K = [[0 for x in range(n)] for y in range(n)]
for i in range(n):
    for j in range(n):
        K[i][j] = Kern(z[i],z[j])
#We then populate the k vector using a for loop.
k = [0 for i in range(n)]
for i in range(n):
    k[i]=Kern(z[i],z[i])
k = np.transpose(np.array(k)[np.newaxis])
#We then determine the H matrix as described by the formula and determine its inverse.
H = (K+((1/(2*c))*np.identity(n)))
Hi = LA.inv(H)

e = np.transpose(np.full((1,n),1))
et = np.full((1,n),1)
#Using these pieces, we can calculate the alphas. Due to the complexity of the formula, we calculate it a piece at a time.
a1 = 2 - np.matmul(np.matmul(et,Hi),k)
a2 = np.matmul(np.matmul(et,Hi),e)
a3 = k + (a1/a2)*e
alphas = np.matmul(.5*Hi,a3)
#We then print the alphas as shown below
print("The Alphas are:\n",np.transpose(alphas))
#With the following as output:
#The Alphas are:
# [[ 0.00645502  0.0023077   0.00311016  0.10934588  0.1695409   0.16417346
#   0.10401649 -0.01186092 -0.00651898 -0.01585425 -0.00421562 -0.02983572
#   0.02222413  0.00309871  0.06654063  0.01966202  0.13360577  0.12568375
#   0.00225207  0.13626879]]
#Using the alphas, we can calculate the radius of the hypersphere. We first calculate the radius of each training point, and then store it in the array R. We then determine the average of the radii.
#To start, we define the array R and initialize it with zeros.
R = [0 for i in range(n)]
for s in range(n):
    R1 = Kern(z[s],z[s])
    r2 = [0 for i in range(n)]
    for j in range(n):
        r2[j] = alphas[j]*Kern(z[s],z[j])
    R2 = -2*sum(r2)
    r3 = [0 for i in range(n**2)]
    for j in range(n):
        for l in range(n):
            r3[(n*j)+l] = alphas[j]*alphas[l]*Kern(z[j],z[l])
    R3 = sum(r3)
    R[s] = R1 + R2 + R3
#The radius of the classifying hypersphere is the mean of the s radii. 
R_sq = np.mean(R)
print("\nR Squared is",R_sq,"\n")
#R Squared is 0.6803321335073261

#Problem 2, Part II: Testing the LS-SVDD
def dz(z_test):
#We then begin a for loop. The first piece of the distance of  D1 is the Kernel of the test vector and itself.
    D1 = Kern(z_test,z_test)
    d2 = [0 for i in range(n)]
    for j in range(n):
        d2[j] = alphas[j]*Kern(z_test,z[j])
    D2 = -2*sum(d2)
    d3 = [0 for i in range(n**2)]
    for j in range(n):
        for l in range(n):
            d3[(n*j)+l] = (alphas[j]*alphas[l])*Kern(z[j],z[l])
    D3 = sum(d3)
    if D1+D2+D3 <= np.sqrt(R_sq):
        return(1)
    else:
        return(0)
#We then feed the New s to the dz function to determine the specificity of the SVM. We know they are not in the target class, we can see how well it correctly classifies a test vector as not being in the class. We initialize an integer sum1 equal to zero. For each test vector it identifies as being the target class, we increment sum1 by 1. If we divide sum1 by the number of test observations, we know the false positive rate, . The complement is the specificity, . 
sum1 = 0
for i in range(len(new_z)):
    print("The Test Value",i,"is",dz(new_z[i]))
    sum1 = sum1 + dz(new_z[i])
print("\nThe Test Specificity of the Model is",1-(sum1/len(new_z)))
#With the following output:
#The Test Value 0 is 1
#The Test Value 1 is 1
#The Test Value 2 is 1
#The Test Value 3 is 0
#The Test Value 4 is 0
#The Test Value 5 is 0
#The Test Value 6 is 1
#The Test Value 7 is 0
#The Test Value 8 is 0
#The Test Value 9 is 0

#The Test Specificity of the Model is 0.6

#Problem 2, Part III: Plotting the LS-SVDD
#We now plot the boundary determined by the LS-SVDD. We will do this by classifying 9,900 points surrounding the training data. We start by defining the arrays containing the points of interest.
x1 = np.array(np.arange(-4,7,.1))
x2 = np.array(np.arange(-5,4,.1))
#We then define the array p, which will contain the points of interest, as well as the predicted outcome. We will also force is to be of the type float.
p = [[0 for x in range(3)] for y in range(len(x2)*len(x1))]
p = np.array(p, dtype=float)
#The array p is populated using nested for loops.
for j in range(len(x2)):
    for i in range(len(x1)):
        p[(len(x1)*j)+i][0] = x1[i]
        p[(len(x1)*j)+i][1] = x2[j]
        p[(len(x1)*j)+i][2] = dz(np.array([x1[i],x2[j]]))
#In order to view these calculated points simultaneously with the training data, we will concatenate the Original and New s and s to p. However, we will change the “classification” of the Original s to 20, and the New s to 10 to highlight their presence on the graph.
for i in range(len(zy)):
    zy[i][2] = 20
for i in range(len(new_zy)):
    new_zy[i][2] = 10

p = np.concatenate((p,zy), axis=0)
p = np.concatenate((p,new_zy), axis=0)
The 9,900 point are then plotted. Yellow points are Original data, blue dots are new data. The light purple area designates points the SVDD determines to be in the target class, while dark purple is outside the target class. 
plt.scatter(p[:,0],p[:,1],c=p[:,2])
plt.xlabel("z_1")
plt.ylabel("z_2", rotation=0)
plt.title("c =%.2f, sigma=%.2f" % (c,sig))       

#Problem 3: Accept/Reject for a Positive Normal Sample

#We now use Python to implement the A/R algorithm. We import numpy for its functions and matplotlib for its plotting functions which we will use to show our algorithm is successful. 
import numpy as np
import matplotlib.pyplot as plt
#We then define our constant C.
C = (np.sqrt(2/np.pi))*np.exp(.5)
#We define the pdf of the positive normal distribution as described above, using an if statement to implement the indicator function.
def pos_pdf(x):
    if x < 0:
        return(0)
    else:
        return(np.sqrt(2/np.pi)*np.exp(-(x**2)/2))
#We then define function posnorm_from_expo, which takes in the size parameter R.
def posnorm_from_expo(R):
#The function defines the counter i, which counts the number of successfully generated samples, and the list pnormal, which will be populated with the R samples.
    i = 0
    pnormal = []
#It then begins a while loop, which will run until all R samples are successfully generated.
    while i < R:
#Inside the loop, we generate an observation u1 from , which is passed to the inverse cumulative distribution function of the exponential distribution.
#This generates a sample from the exponential distribution, y.
        u1 = float(np.random.uniform(0,1,1))
        y = -np.log(1 - u1)
        u2 = np.random.uniform(0,1,1)
        check = pos_pdf(y)/(C*(np.exp(-y)))
        if ( u2 < check ):
            i = i + 1
            pnormal.append(y)
#We then return the sample of size R.
    return(pnormal)
#To show our algorithm is successful, we plot a histogram of the sample against the curve of the positive normal. We start by making an array p of values from 0 to 5 in increments of .5. We then define an array of zeros pc which we then populate with the value of the positive normal pdf from the values in p using a for loop.
p = np.array(np.arange(0,5,.05)) 
pc = [0 for i in range(len(p))]
for i in range(len(p)):
    pc[i] = pos_pdf(p[i])
#We then simulate 100,000 observations using the A/R algorithm, plot the histogram, and then plot the Positive Normal pdf against it.
sim = posnorm_from_expo(100000)
plt.hist(sim, bins='auto',density=1)
plt.title("Positive Normal Simulation")
h = plt.plot(p, pc, lw=2)
plt.show()
