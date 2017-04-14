% package: bayesCL
\name{prepare}
\alias{prepare}

\title{GPU preparation for PolyaGamma sampling and/or Bayesian Inference}

\description{
  Generates the external pointer to the GPU. This function compiles the OpenCL code, creates the command queue, etc. It can be used in order to avoid compilation/creation in each call of the rpg, mlr, and lasso.
}

\details{

  This is used in order to avoid unnecesarry recompilation of OpenCL kernel and creation of contexts, command queues, etc.. The output of this function is a pointer that can be passed to the mlr, lasso and rpg functions. If the pointer is not passed to these functions, the prepare function is called from inside the mlr/lasso/rpg functions in each call. If no device number is specified, a list of devices with their respective IDs will be shown and you will be prompted to enter a number. In order to tune the implementation you can specify your own values for implementation parameters, which is a 9 dimensional vector.

}

\usage{

prepare(precision=0,device=-1, parameters=NULL )

}

\arguments{

  \item{precision}{the number of random variates to simulate.}
  \item{device}{the ID of the device for which to generate the helper variables.}
  \item{parameters}{a 9 dimensional vector of parameters to tune the GPU implementation.}
  
}

\value{

  This function returns an external pointer to a C structure for the GPU.

}

\examples{
  gpu_pointer <- prepare(precision=0, device=0)
}

\seealso{
  \code{\link{rpg},\link{lasso},\link{mlr}}
}


\keyword{GPU}
