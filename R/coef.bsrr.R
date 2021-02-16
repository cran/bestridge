#' Provides estimated coefficients from a fitted "bsrr" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{bsrr}" object.
#'
#'
#' @param object A "\code{bsrr}" project.
#' @param sparse Logical or NULL, specifying whether the coefficients should be
#' presented as sparse matrix or not.
#' @param \dots Other arguments.
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bsrr}}, \code{\link{print.bsrr}}.
#' @references Wen, C., Zhang, A., Quan, S. and Wang, X. (2020). BeSS: An R
#' Package for Best Subset Selection in Linear, Logistic and Cox Proportional
#' Hazards Models, \emph{Journal of Statistical Software}, Vol. 94(4).
#' doi:10.18637/jss.v094.i04.
#' @return
#' If \code{sparse == FALSE}, a vector containing the estimated coefficients
#' from a fitted "\code{bsrr}" object is returned. If \code{sparse == TRUE},
#' a \code{dgCMatrix} containing the estimated coefficients
#' is returned.
#' @examples
#'
#' # Generate simulated data
#' n <- 200
#' p <- 20
#' k <- 5
#' rho <- 0.4
#' seed <- 10
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bsrr(Data$x, Data$y, method = "pgsection")
#' coef(lm.bsrr)
#' @export
#'
#'
coef.bsrr <- function(object, sparse=TRUE, ...)
{
  if(!is.null(object$coef0)){
    beta<-c(intercept=object$coef0, object$beta)
    names(beta)[1] <- "(intercept)"
  } else beta<-object$beta
  if(sparse==TRUE)
  {
    beta<-matrix(beta,byrow =TRUE, dimnames = list(names(beta)))
    beta<-Matrix(beta,sparse = TRUE)
    return(beta)
  }else return(beta)
}
