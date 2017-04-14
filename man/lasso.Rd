% package: bayesCL
\name{lasso}
\alias{lasso}
\title{Bayesian Lasso}

\description{
  Inference for Bayesian lasso regression models by Gibbs sampling from the Bayesian posterior 
  distribution.
}

\details{

  The Bayesian lasso model, hyperprior for the lasso parameter, and Gibbs Sampling algorithm implemented by this function are identical to that is described in detail in Park & Casella (2008). The GPU implementation is derived from the CPU implementation blasso from package monomvn.

}

\usage{

lasso(X, y, T=1000, lambda2=1, beta = NULL, s2=var(y-mean(y)),
           rd=NULL, ab=NULL, icept=TRUE,
           normalize=TRUE, device=0, parameters=NULL)

}

\arguments{

 \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{y}{ vector of output responses \code{y} of length equal to the
    leading dimension (rows) of \code{X}, i.e., \code{length(y) == nrow(X)}}
  \item{T}{ total number of MCMC samples to be collected }
  \item{beta}{ initial setting of the regression coefficients.}
  \item{lambda2}{ square of the initial lasso penalty parameter. }
  \item{s2}{ initial variance parameter. }
  \item{rd}{ \code{=c(r, delta)}, the alpha (shape) parameter and
    \eqn{\beta}{beta} (rate) parameter to the gamma distribution prior
    \code{G(r,delta)} for the \eqn{\lambda^2}{lambda2} parameter under
    the lasso model. A default of \code{NULL}
   generates appropriate non-informative values depending on the
   nature of the regression. }
 \item{ab}{ \code{=c(a, b)}, the \eqn{\alpha}{alpha} (shape)
   parameter and the \eqn{\beta}{beta} (scale) parameter for the
   inverse-gamma distribution prior \code{IG(a,b)} for the variance
   parameter \code{s2}.  A default of \code{NULL} generates appropriate
   non-informative values depending on the nature of the regression. }
  \item{icept}{ if \code{TRUE}, an implicit intercept term is fit
    in the model, otherwise the the intercept is zero; default is
    \code{TRUE}. }
  \item{normalize}{ if \code{TRUE}, each variable is standardized
    to have unit L2-norm, otherwise it is left alone; default is
    \code{TRUE}. }
  \item{device}{If no external pointer is provided to function, we can provide the ID of the device to use.}
  \item{parameters}{a 9 dimensional vector of parameters to tune the GPU implementation. }

}

\value{

  \code{lasso} returns an object of class \code{"lasso"}, which is a
  \code{list} containing a copy of all of the input arguments as well as
  of the components listed below.
  
  \item{mu }{ a vector of \code{T} samples of the (un-penalized)
    \dQuote{intercept} parameter. }
  \item{beta }{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (penalized) regression coefficients.}
  \item{s2 }{ a vector of \code{T} samples of the variance parameter}
  \item{lambda2 }{ a vector of \code{T} samples of the penalty
    parameter.}
  \item{tau2i}{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (latent) inverse diagonal of the prior covariance matrix for
    \code{beta}, obtained for Lasso regressions. }
}

\seealso{
  \code{\link{rpg},\link{mlr}}
}

\examples{

set.seed(0)
n_samples  <- 500
n_features <- 40
X <- matrix(rnorm(n_features * n_samples), nrow = n_samples)
y <- 2 * X[,1] - 3 * X[,2] + rnorm(n_samples) # only features 1 & 2 are relevant

X_train <- X[1:400,]
y_train <- y[1:400]
X_test  <- X[401:500,]
y_test  <- y[401:500]


# START ------------------------------------------------------------------------

# first, standardize data !!!
X_train <- scale(X_train)

tmp00 <- bayesCL::lasso(X = X_train, 
                          y = y_train, 
                          T = 500,  # number of Gibbs sampling iterations
                          icept = T,
                          device=0  ) # use constant term (intercept), we do


#scale test data based on train data means and scales!!
X_test <- scale(X_test,
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:scale"))


p_train1 <- colMeans(tmp00$beta \%*\% t(X_train))
p_test1 <- colMeans(tmp00$beta \%*\% t(X_test))

plot(y_train, p_train1, col = "red", xlab = "actual", ylab = "predicted")
points(y_test, p_test1, col = "green")



}
\keyword{ regression }
