% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/coef.bsrr.R
\name{coef.bsrr}
\alias{coef.bsrr}
\title{Provides estimated coefficients from a fitted "bsrr" object.}
\usage{
\method{coef}{bsrr}(object, sparse = TRUE, ...)
}
\arguments{
\item{object}{A "\code{bsrr}" project.}

\item{sparse}{Logical or NULL, specifying whether the coefficients should be
presented as sparse matrix or not.}

\item{\dots}{Other arguments.}
}
\value{
If \code{sparse == FALSE}, a vector containing the estimated coefficients
from a fitted "\code{bsrr}" object is returned. If \code{sparse == TRUE},
a \code{dgCMatrix} containing the estimated coefficients
is returned.
}
\description{
This function provides estimated
coefficients from a fitted "\code{bsrr}" object.
}
\examples{

# Generate simulated data
n <- 200
p <- 20
k <- 5
rho <- 0.4
seed <- 10
Tbeta <- rep(0, p)
Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
lm.bsrr <- bsrr(Data$x, Data$y, method = "pgsection")
coef(lm.bsrr)
}
\seealso{
\code{\link{bsrr}}, \code{\link{print.bsrr}}.
}
\author{
Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu, Canhong Wen and Xueqin Wang.
}
