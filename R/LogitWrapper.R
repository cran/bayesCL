## Draw PG(n, z) where n is a natural number.
##------------------------------------------------------------------------------

rpg <- function( num=1, n=1, z=0.0, batch=32, local=128, staticseed=FALSE, seed=0, float=0, ptr=NULL, device=0)
{
    ## Check Parameters.
    if (any(n<0)) {
      print("n must be greater than zero.");
      return(NA);
    }
    if(is.null(ptr)){
      ptr<-prepare(float, device=device)
    }
    ## This is a quick hack to find the library address
    ## will need to save it somewhere
    x = rep(0, num);
    
    if (length(n) != num) { n = array(n, num); }
    if (length(z) != num) { z = array(z, num); }
    if(staticseed==FALSE){
     static_seed=0;
    }else{
     static_seed=1;
    }
    if(float==0){
      OUT <-.Call("rpg_devroye", ptr, as.integer(batch), as.integer(n), as.single(z), as.integer(num), as.integer(local), as.integer(static_seed) , as.integer(seed));  
    }else{
      OUT <-.Call("rpg_devroye_double", ptr, as.integer(batch), as.integer(n), as.single(z), as.integer(num), as.integer(local), as.integer(static_seed) , as.integer(seed));
    }
    
    return(OUT);
}

prepare <- function(precision=0,device=-1, parameters=NULL){
  cat("------------------------------\n")
  cat("Platform and device selection\n")
  platforms<-.Call("find_platforms")
  pls<-c()
  devs<-c()
  i<-0;
  j<-0;
  if(device==-1 ){
    for(p in platforms){
      devices<-.Call("find_devices",as.integer(i))
      k<-0
      for(d in devices){
        cat(sprintf("  %d - %s on platform %s\n",j,d,p))
        pls<-c(pls,i)
        devs<-c(devs,k)
        j=j+1
        k<-k+1
      }
      i<-i+1
    }
    cat("\n")
    if(j==1){
      cat("Only 1 device found. No selection needed. ")
      device<-0
    }else{
      ans <- readline(prompt="Select a device: ")
      device<-as.integer(ans)
      if(device>i)
        stop("Entered the wrong device ID!")
    }
  }else{
    dvs<-c()
    for(p in platforms){
      devices<-.Call("find_devices",as.integer(i))
      k<-0
      for(d in devices){
        sprintf("  %d - %s on platform %s\n",j,d,p)
        dvs<-c(dvs,d)
        pls<-c(pls,i)
        devs<-c(devs,k)
        j=j+1
        k<-k+1
      }
      i<-i+1
    }
    if(device>i)
      stop("Entered the wrong device ID!")
    else
      cat(sprintf("Selected device %s on platform %s\n",dvs[device+1],platforms[pls[device+1]+1]));
  }
  platf<-pls[device+1]
  device_id<-devs[device+1]
  if(is.null(parameters)){
    params<-c(30,128,32,16,16,-4,32,32,8)    
  }else{
    params<-parameters
  }
  if(precision==0)
  {
    openclptrs<-.Call("prepareCL", as.integer(platf),as.integer(device_id), as.character(paste(.libPaths()[1],"/bayesCL",.Platform$file.sep,"polyaGammaf.cl",sep="")), as.character(paste(.libPaths()[1],"/bayesCL",.Platform$file.sep,"polyaGammaf-",platf,"-",device_id,".bin",sep="")), as.integer(params));    
  } else{
    openclptrs<-.Call("prepareCL", as.integer(platf),as.integer(device_id), as.character(paste(.libPaths()[1],"/bayesCL",.Platform$file.sep,"polyaGamma.cl",sep="")), as.character(paste(.libPaths()[1],"/bayesCL",.Platform$file.sep,"polyaGamma-",platf,"-",device_id,".bin",sep="")), as.integer(params));    
  }
  return(openclptrs)
}

mult.check.parameters <- function(y, X, n, m.0, P.0, samp, burn)
{
    ok = rep(TRUE, 6);
    ok[1] = all(y >= 0);
    ok[2] = all(n > 0);
    ok[3] = (nrow(y) == length(n) && nrow(y) == nrow(X));
    ok[4] = (samp > 0);
    ok[5] = (burn >=0);
    ok[6] = all(rowSums(y) <= 1);
    ok[7] = (ncol(y)==ncol(m.0) && ncol(X)==nrow(m.0));
    ok[8] = (ncol(X)==dim(P.0)[1] && ncol(X)==dim(P.0)[2] && ncol(y)==dim(P.0)[3]);

    if (!ok[1]) print("y must be >= 0.");
    if (!ok[6]) print("y[i,] are proportions and must sum <= 1.");
    if (!ok[2]) print("n must be > 0.");
    if (!ok[3]) print(paste("Dimensions do not conform for y, X, and n.",
                            "dim(y) =", nrow(y), ncol(y),
                            "dim(x) =", nrow(X), ncol(X),
                            "len(n) =", length(n)));
    if (!ok[4]) print("samp must be > 0.");
    if (!ok[5]) print("burn must be >=0.");
    if (!ok[7]) print("m.0 does not conform.");
    if (!ok[8]) print("P.0 does not conform.");

    ok = all(ok)
}


## Posterior for multinomial logistic regression
##------------------------------------------------------------------------------
mlr <- function(y, X, n=rep(1,nrow(as.matrix(y))),
                   m.0=array(0, dim=c(ncol(X), ncol(y))),
                   P.0=array(diag(0, ncol(X)), dim=c(ncol(X),ncol(X),ncol(y))),
                   samp=1000, burn=500, float=0, device=0, parameters=NULL )
{
    ## In the event y or X is one dimensional.
    X = as.matrix(X);
    y = as.matrix(y);

    pointer<-prepare(precision<-float, device<-device, parameters<-parameters)  
    
    N = dim(X)[1];
    P = dim(X)[2];
    J = dim(y)[2]+1;

    ## Check that the data and priors are okay.
    ok = mult.check.parameters(y, X, n, m.0, P.0, samp, burn);
    if (!ok) return(NA)

    ## Initialize output.
    output = list();

    beta = array(0.0, dim=c(P, J-1, samp));
    timing = array(2.0, dim=c(6)) #timing vector (all, computing eta and c, pg draw, computing the p matrix and b vector, cholesky to compute beta + calculate the new Xb)
    ## Our Logit function, written in C, uses t(X), t(y).
    tX = t(X);
    ty = t(y);
    if(float==0){
      OUT = .Call("mult_gibbs", beta, as.double(ty), as.double(tX), as.double(n), as.double(m.0), as.double(P.0), as.integer(N), as.integer(P), as.integer(J), as.integer(samp), as.integer(burn), as.double(timing),pointer);
    }else{
      OUT = .Call("mult_gibbs_double", beta, as.double(ty), as.double(tX), as.double(n), as.double(m.0), as.double(P.0), as.integer(N), as.integer(P), as.integer(J), as.integer(samp), as.integer(burn), as.double(timing),pointer);
    }
    N = OUT$N;

    # #Transpose for standard output.
    beta = array(0, dim=c(samp, P, J-1));
    for (i in 1:samp) {
      beta[i,,] = OUT$beta[,,i]
    }
    for (i in 1:6){
      timing[i]=OUT$timing[i]
     }

    output = list("beta"=beta, "y"=y, "X"=X, "n"=n, "timing"=timing);
    
    output
}


'lasso' <-
  function(X, y, T=1000, lambda2=1, beta = NULL, s2=var(y-mean(y)),
           rd=NULL, ab=NULL, icept=TRUE,
           normalize=TRUE, device=0, parameters=NULL)
  {
    
    ## dimensions of the inputs
    X <- as.matrix(X)
    m <- ncol(X)
    n <- nrow(X)
    timing<-c(0.0,0.0,0.0,0.0)

    cl <- match.call()
    y <- as.numeric(y)
    
    if(length(y) != nrow(X))
      stop("must have nrow(X) == length(y)")
    
    ## check T
    if(length(T) != 1 || T <= 1)
      stop("T must be a scalar integer > 1")
    
    ## check lambda2
    if(length(lambda2) != 1 || lambda2 <= 0)
      stop("lambda2 must be a positive scalar")
    
    lambda2 <- as.double(rep(lambda2, T))
    
    thin<-1
    
    ## check s2
    if(length(s2) != 1 || s2 <= 0)
      stop("s2 must be a positive scalar")
    
    ## check tau2i or default
    
    tau2i <- rep(1, m)
    tau2i <- as.double(rep(tau2i, T))
    
    ## check r and delta (rd)
    if(is.null(rd)) {
      rd <- c(2, 0.1) ## otherwise lasso G prior
    }
    ## double-check rd
    if(length(rd) != 2 || any(rd <= 0))
      stop("rd prior must be a positive 2-vector")
    
    if(is.null(beta))
      beta <- rep(1, m)
    
    if(length(beta) != m )
      stop("beta must be of length number of parameters")
    T1<-double(T)
    
    ## check ab or default
    if(is.null(ab))
    {
      ab <- c(0,0)
      if(m >= n)
      {
        ab[1] <- 3/2
        ab[2] <-  0.1759232*sum(y^2)
      }
    }
    
    ## double check ab
    if(length(ab) != 2 || any(ab < 0))
      stop("ab must be a non-negative 2-vector")
    
    ## check normalize
    if(length(normalize) != 1 || !is.logical(normalize))
      stop("normalize must be a scalar logical")
    
    pointer<-prepare(precision=0, device=device, parameters = parameters  )
    
    
    r <- .Call("blassoGPU",as.integer(T),as.integer(thin),as.integer(m),as.integer(n),as.double(t(X)),as.double(y),as.double(lambda2),as.double(T1),as.double(rep(beta, T)),as.double(rep(s2, T)),as.double(tau2i),as.double(rd[1]),as.double(rd[2]),as.double(ab[1]),as.double(ab[2]),as.integer(normalize),as.double(timing),pointer)
    
    
    #turn the beta vector of samples into a matrix
    r$beta <- matrix(r$beta, nrow=T, ncol=m, byrow=TRUE, dimnames=list(NULL,paste("b.", 1:m, sep="")))


    # turn the tau2i vector of samples into a matrix
    if(r$lambda2[1] != 0 && length(r$tau2i) > 0)
    {
      r$tau2i <- matrix(r$tau2i, nrow=T, ncol=m, byrow=TRUE, dimnames=list(NULL,paste("tau2i.", 1:m, sep="")))
      ## put NAs where tau2i has -1
      r$tau2i[r$tau2i == -1] <- NA
    }
    else if(length(r$tau2i) > 0)
    {
      r$lambda <- r$tau2i <- NULL
    }
    else
    {
      r$tau2i <- NULL
    }
  
    ## null-out redundancies
    r$lambda2.len <- r$tau2i.len <- NULL
    if(length(r$lambda2) == 0) r$lambda2 <- NULL

    
    return(r)
  }