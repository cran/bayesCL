% package: bayesCL
\name{mlr}

\alias{mlr}

\title{Bayesian Multinomial Logistic Regression}

\description{
  Inference for Bayesian multinomial logistic regression models by Gibbs sampling from the Bayesian posterior 
  distribution.
}

\details{

  Classic multinomial logistic regression for classifiction.

  We assume that \eqn{\beta_J = 0} for purposes of identification.

}

\usage{

mlr(y, X, n=rep(1,nrow(as.matrix(y))),
                   m.0=array(0, dim=c(ncol(X), ncol(y))),
                   P.0=array(diag(0, ncol(X)), dim=c(ncol(X),ncol(X),ncol(y))),
                   samp=1000, burn=500, float=0, device=0, parameters=NULL)

}


\arguments{
  \item{y}{an N x J-1 dimensional matrix; 
  	   \eqn{y_{ij}} is the average response for category j at \eqn{x_i}.}
  \item{X}{an N x P dimensional design matrix; \eqn{x_i} is the ith row.}
  \item{n}{an N dimensional vector; \eqn{n_i} is the total number of observations at each \eqn{x_i}.}
  \item{m.0}{a P x J-1 matrix with the \eqn{\beta_j}'s prior means.}
  \item{P.0}{a P x P x J-1 array of matrices with the \eqn{\beta_j}'s prior precisions.}
  \item{samp}{the number of MCMC iterations saved.}
  \item{burn}{the number of MCMC iterations discarded.}
  \item{float}{a number representing the degree of precision to use: for single-precision floating point use 0, for or double-precision floating point use 1. }
  \item{device}{if no external pointer is provided to function, we can provide the ID of the device to use.}
  \item{parameters}{a 9 dimensional vector of parameters to tune the GPU implementation. }
  
}

\value{

  \code{mlr} returns a list.

  \item{beta}{a samp x P x J-1 array; the posterior sample of the regression
  coefficients.}

  \item{w}{a samp x N' x J-1 array; the posterior sample of the latent variable.
  WARNING: N' may be less than N if data is combined.}

  \item{y}{the response matrix--different than input if data is combined.}

  \item{X}{the design matrix--different than input if data is combined.}

  \item{n}{the number of samples at each observation--different than input if
  data is combined.}
  
  
}

\examples{

## Use the iris dataset.
data(iris)
N <- nrow(iris)
P <- ncol(iris)
J <- nlevels(iris$Species)

X     <- model.matrix(Species ~ ., data=iris);
y.all <- model.matrix(~ Species - 1, data=iris);
y     <- y.all[,-J];

out <- mlr(y, X, samp=1000, burn=100, device=0);

}

\seealso{
\code{\link{rpg},\link{lasso}}
}


\keyword{mlr}
\keyword{regression}