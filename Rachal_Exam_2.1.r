data = read.csv(file = "charlie.csv")

Data = as.matrix(data,ncol=8)

Y = Data[,1]
X = as.matrix(Data[,3:6])
X = matrix(as.numeric(unlist(X)),nrow=nrow(X))
n = length(Y)

for (i in 1:n){
  if (Y[i] == "Original"){
    Y[i] = 1
  }
  else{
    Y[i] = -1
  }
}
Y = as.integer(Y)

#### Functions ####

rbf_kernel = function(x1,x2,gamma){
  K = exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(K)
}

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

H_func = function(K,c){
  
  n = length(K[,1])
  H = K + c*diag(n)

  return(H)
  
}

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

alphas(Data,3,.01)

H_inv_up = function(data,old_H,xn,c,gamma){
  
  n = length(old_H[,1])
  
  kn = as.matrix(rep(0,n))
  
  for (i in 1:n){
    kn[i] = rbf_kernel(xn,data[i,],gamma)
    }
  
  an = old_H %*% kn
  
  knn = rbf_kernel(xn,xn,gamma)
  
  gamman = as.double(knn + c - (t(kn) %*% an))
  
  side1 = rbind((gamman*old_H) + (an %*% t(an)),t(-an))
  
  side2 = rbind(-an,c(1))
  
  H_inv_u = (1/gamman)*cbind(side1,side2)
  
  return(H_inv_u)

  }

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

#### Testing Alphas #### 

Hi4 = solve(H_func(K_func(X4,3)$K,.01))
Hi5 = solve(H_func(K_func(X5,3)$K,.01))
Hi6 = solve(H_func(K_func(X6,3)$K,.01))


X4 = X[(1:4),]
X5 = X[(1:5),]
X6 = X[(1:6),]
X7 = X[(1:7),]

x5 = X[5,]
x6 = X[6,]
x7 = X[7,]

alpha5 = alphas(X5,3,.01)
alpha5_u = alpha_up(H_inv_up(X4,Hi4,x5,.01,3))

alpha6 = alphas(X6,3,.01)
alpha6_u = alpha_up(H_inv_up(X5,Hi5,x6,.01,3))

alpha7 = alphas(X7,3,.01)
alpha7_u = alpha_up(H_inv_up(X6,Hi6,x7,.01,3))

#### Distance Functions ####

distance = function(z,data,alphas,gamma){
  
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

R_sq = function(data,alphas,gamma){
  
  n = length(data[,1])
  
  r = rep(0,n)
  
  for ( i in 1:n ){
    r[i] = distance(data[i,],data,alphas,gamma)
  }
  
  return(mean(r))
  
}

#### Distance Testing ####

alpha4 = alphas(X4,3,.01)
x8 = X[8,]

R4 = R_sq(X4,3)
d5 = distance(x5,X4,alpha4,3)
isTRUE(d5 < R4)

R5 = R_sq(X5,alpha5_u,3)
d6 = distance(x6,X5,alpha5_u,3)
isTRUE(d6 < R5)

R6 = R_sq(X6,alpha6_u,3)
d7 = distance(x7,X6,alpha6_u,3)
isTRUE(d7 < R6)

R7 = R_sq(X7,alpha7_u,3)
d8 = distance(x8,X7,alpha7_u,3)
isTRUE(d6 < R7)