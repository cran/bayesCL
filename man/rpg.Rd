% package: bayesCL
\name{rpg}
\alias{rpg}

\title{Polya-Gamma Random Variates using a GPU}

\description{
  Generate random variates from the Polya-Gamma distribution on a GPU.
}

\details{

  A random variable X with distribution PG(n,z) is distributed like

  \deqn{ X \sim \sum_{k=1}^\infty G(n,1) / ( 2 \pi^2 (k-1/2)^2 + z^2/2). }

  The density for X may be derived by exponentially tilting the
  PG(n,0) density:

  \deqn{ p(x|n,z) \propto \exp(-x z^2/2) p(x|n,0).}

  The GPU implementation is derived from the CPU implementation rpg.devroye from package BayesLogit.
  
}

\usage{

rpg(num=1, n=1, z=0.0, batch=32, local=128, staticseed=FALSE,
seed=0, float=0, ptr=NULL, device=0)

}

\arguments{

  \item{num}{the number of random variates to simulate.}
  \item{n}{shape parameter, a positive integer.}
  \item{z}{parameter associated with tilting.}
  \item{batch}{the number of samples created by each GPU thread}
  \item{local}{the number of threads in a thread-block on a GPU}
  \item{staticseed}{parameter to determine whether to use a static seed or not.}
  \item{seed}{the value of the static seed, if used.}
  \item{float}{parameter to determine whether to use single-precision floating point or double-precision floating point.}
  \item{ptr}{an external pointer to the C structure with the GPU helper variables.}
  \item{device}{if no external pointer is provided to function, we can provide the ID of the device to use.}
  
}

\value{

  This function returns \code{num} Polya-Gamma samples.

}

\examples{
random_variates <- rpg(num=100, n=1, z=0.0, device=0)
}

\seealso{
  \code{\link{prepare},\link{mlr}}
}


\keyword{Polya-Gamma}
\keyword{polyagamma}
