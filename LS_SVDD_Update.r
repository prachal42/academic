#6106 - Statistical Computing 
#Exam 2
#Patrick Rachal
#Problem 1: LS-SVDD Update
#In this problem we continue the theme of support vector machines, and updating their calculations based on new observations. Here we introduce a new method of updating the inverse matrix given a new observation has been added to the dataset. 


#We use this updated matrix to calculate the  required for the distance and radius functions. The formula for  has been discussed in previous reports. The new strategy is implemented in R. 
#In R, we begin by reading the csv file for the Charlie data used in previous projects.
data = read.csv(file = "charlie.csv")
#The data is stored in a matrix for ease of manipulation.
Data = as.matrix(data,ncol=8)
#We separate the data we need, storing the response values in Y and the predictors in X. We then determine the total amount of observations in the dataset.
Y = Data[,1]
X = as.matrix(Data[,3:6])
X = matrix(as.numeric(unlist(X)),nrow=nrow(X))
n = length(Y)
#We then recode the responses into +1 and -1 for the SVM.
for (i in 1:n){
  if (Y[i] == "Original"){
    Y[i] = 1
  }
  else{
    Y[i] = -1
  }
}
#The result is still a string, so we force it into an integer.
Y = as.integer(Y)
#We then begin the functional definitions. We begin with the required Kernel function:
rbf_kernel = function(x1,x2,gamma){
  K = exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(K)
}
#We then define the K function, which takes in the data to be used, and the parameter gamma for the kernel. It returns both the K matrix and the k vector.
K_func = function(data, gamma){
  n = length(data[,1])
  k = rep(0,n)
  K = matrix(0,n,n)
  for (i in 1:n){
    k[i] = rbf_kernel(X[i,],X[i,],gamma)
    for (j in 1:n){
      K[i,j] = rbf_kernel(X[i,],X[j,],gamma)
    }
  }
  list(K=K,k=k)
}
#We then define the H function, which takes in only the K matrix and the parameter c.
H_func = function(K,c){
  n = length(K[,1])
  H = K + c*diag(n)
  return(H)
}
#We then define the alphas function, which takes in the data, and the two parameters gamma and c. The function uses the H function, which calls the K function, which calls the kernel function. The formulation of the function is identical to Python implementation.
alphas = function(data,gamma,c){
  K = K_func(data,gamma=gamma)$K
  k = K_func(data,gamma=gamma)$k
  k = as.matrix(k)
  H = H_func(K,c)
  n = length(H[,1])
  H_inv = solve(H)
  e = as.matrix(rep(1,n))
  a = 2 - ( t(e) %*% H_inv %*% k )
  b = t(e) %*% H_inv %*% e
  c = as.double(a/b)
  d = k + c*e
  return(alpha = .5 * H_inv %*% d)
}
#Part a: Inverse H Function
#We then begin the new work, defining the inverse update function. It requires the original data, the most recent  matrix, the row being added to the data, c and gamma.
H_inv_up = function(data,old_H,xn,c,gamma){
#It first determined the length of the original H, and then fills a matrix full of zeros, which are immediately populated with the kernel values.
  n = length(old_H[,1])
  kn = as.matrix(rep(0,n))
  
  for (i in 1:n){
    kn[i] = rbf_kernel(xn,data[i,],gamma)
    }
  an = old_H %*% kn
  knn = rbf_kernel(xn,xn,gamma)
  gamman = as.double(knn + c - (t(kn) %*% an))
#The updated  matrix is then stitched together via rbind and cbind, and returned by the function.
  side1 = rbind((gamman*old_H) + (an %*% t(an)),t(-an)) 
  side2 = rbind(-an,c(1))
  H_inv_u = (1/gamman)*cbind(side1,side2) 
  return(H_inv_u)
  }
#Part b: Updating Alphas
#We finally define the alpha update function, which is identical to the alphas function except it only requires the inverse H matrix.
alpha_up = function(H_inv){
  n = length(H_inv[,1])
  k = rep(1,n)
  e = rep(1,n)
  a = 2 - ( t(e) %*% H_inv %*% k )
  b = t(e) %*% H_inv %*% e
  c = as.double(a/b)
  d = k + c*e
  return(alpha = .5 * H_inv %*% d)
}
#Part c: Checking the Work
#We then proceed to check the functions work correctly. We first calculate ,,.
Hi4 = solve(H_func(K_func(X4,3)$K,.01))
Hi5 = solve(H_func(K_func(X5,3)$K,.01))
Hi6 = solve(H_func(K_func(X6,3)$K,.01))
#We then extract the first four, five, six, and seven rows separately.
X4 = X[(1:4),]
X5 = X[(1:5),]
X6 = X[(1:6),]
X7 = X[(1:7),]
#And we extract the fifth, sixth, and seventh rows individually.
x5 = X[5,]
x6 = X[6,]
x7 = X[7,]
#We use the alphas function to calculate the alphas off the raw data.
alpha5 = alphas(X5,3,.01)

alpha5_u = alpha_up(H_inv_up(X4,Hi4,x5,.01,3))
#The results are identical.
  
#We repeat the same process, adding the sixth and seventh rows.
alpha6 = alphas(X6,3,.01)
alpha6_u = alpha_up(H_inv_up(X5,Hi5,x6,.01,3))
  
alpha7 = alphas(X7,3,.01)
alpha7_u = alpha_up(H_inv_up(X6,Hi6,x7,.01,3))
  
#Part d: Distance Function and 
#Here we use the updated alphas to sequentially update the LS SVDD and check to see if the proceeding observations are outliers. The formulation of the distance function and  has been explored in previous reports. The distance function works piecewise, constructing the individual parts of the sum before finally adding them all together and returning the distance. It requires the observation whose distance from the center is being determined, the data whose center it is that we care about, and finally the gamma parameter.
distance = function(z,data,alphas,gamma,c){
  n = length(data[,1])
  A = rbf_kernel(z,z,gamma)
  B = rep(0,n)
  for ( i in 1:n){
    B[i] = alphas[i]*rbf_kernel(z,data[i,],gamma) 
  }
  B = -2*sum(B)
  C = matrix(0,n,n)
  for (i in 1:n){
    for (j in 1:n){
      C[i,j] = alphas[i]*alphas[j]*rbf_kernel(data[i,],data[j,],gamma)
    }
  }
  C = sum(C)
  return (A+B+C)
}
#The R_sq function works by calling the distance function for all the rows in the data and averaging their distances from the center. The function then returns the result.
R_sq = function(data,alphas,gamma,c){
  n = length(data[,1])
  r = rep(0,n)
  for ( i in 1:n ){
    r[i] = distance(data[i,],data,alphas,gamma,c)
  }
  return(mean(r))
}
#We then test our results. We first calculate the alphas corresponding to the first four observations, and the eighth row of data.
alpha4 = alphas(X4,3,.01)
x8 = X[8,]
#We then calculate the radius of the first four observations and the distance of the fifth observation from the first four. We then check to see if the distance is less than the radius.
R4 = R_sq(X4,3,.01)
d5 = distance(x5,X4,3,.01)
isTRUE(d5 < R4)
#[1] FALSE
#So the fifth row is an outlier compared to the previous data. We then proceed with the rest of the observations, adding them to the radius calculation and then checking the next.  All are considered outliers relative to their proceeding data.
R5 = R_sq(X5,alpha5_u,3,.01)
d6 = distance(x6,X5,alpha5_u,3,.01)
isTRUE(d6 < R5)
#[1] FALSE
R6 = R_sq(X6,alpha6_u,3,.01)
d7 = distance(x7,X6,alpha6_u,3,.01)
isTRUE(d7 < R6)
#[1] FALSE
R7 = R_sq(X7,alpha7_u,3)
d8 = distance(x8,X7,alpha7_u,3)
isTRUE(d6 < R7)
#[1] FALSE
