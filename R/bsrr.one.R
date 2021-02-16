#' Best subset ridge regression with a
#' specified model size and a shrinkage parameter
#'
#' Best subset ridge regression with a specified model size for generalized
#' linear models and Cox's proportional hazard model.
#'
#'  Given a model size \eqn{s} and a shrinkage parameter \eqn{\lambda}
#', we consider the following best subset ridge regression problem:
#'\deqn{\min_\beta -2 \log L(\beta) + \lambda \Vert\beta \Vert_2^2; { s.t.} \|\beta\|_0 = s.}
#'
#' The best subset selection problem is a special case of the best ridge regression problem
#' with the shrinkage \eqn{\lambda=0}.
#'
#'In the GLM case, \eqn{\log L(\beta)} is the log likelihood function;
#' In the Cox model, \eqn{\log L(\beta)} is the log partial likelihood function.
#'
#'The best ridge regression is solved by the \eqn{L_2} penalized primal dual active set algorithm
#'for best ridge regression. This algorithm utilizes an active set updating strategy
#'via primal and dual variables and fits the sub-model by exploiting the fact that their
#' support set are non-overlap and complementary.
#'
#' @param x Input matrix, of dimension \eqn{n \times p}; each row is an observation
#' vector and each column is a predictor/feature/variable.
#' @param y The response variable, of \code{n} observations. For \code{family = "binomial"} should be
#' a factor with two levels. For \code{family="poisson"}, \code{y} should be a vector with positive integer.
#'  For \code{family = "cox"}, \code{y} should be a two-column matrix
#' with columns named \code{time} and \code{status}.
#' @param family One of the following models: \code{"gaussian"}, \code{"binomial"},
#' \code{"poisson"}, or \code{"cox"}. Depending on the response.
#' @param s A specified model size
#' @param lambda A shrinkage parameter for \code{"bsrr"}.
#' @param always.include An integer vector containing the indexes of variables that should always be included in the model.
#' @param screening.num Users can pre-exclude some irrelevant variables according to maximum marginal likelihood estimators before fitting a
#' model by passing an integer to \code{screening.num} and the sure independence screening will choose a set of variables of this size.
#' Then the active set updates are restricted on this subset.
#' @param normalize Options for normalization. \code{normalize = 0} for
#' no normalization. Setting \code{normalize = 1} will
#' only subtract the mean of columns of \code{x}.
#' \code{normalize = 2} for scaling the columns of \code{x} to have \eqn{\sqrt n} norm.
#' \code{normalize = 3} for subtracting the means of the columns of \code{x} and \code{y}, and also
#' normalizing the columns of \code{x} to have \eqn{\sqrt n} norm.
#' If \code{normalize = NULL}, by default, \code{normalize} will be set \code{1} for \code{"gaussian"},
#' \code{2} for \code{"binomial"} and \code{"poisson"}, \code{3} for \code{"cox"}.
#' @param weight Observation weights. Default is \code{1} for each observation.
#' @param max.iter The maximum number of iterations in the \code{bsrr} function.
#' In most of the case, only a few steps can guarantee the convergence. Default
#' is \code{20}.
#' @param group.index A vector of integers indicating the which group each variable is in.
#' For variables in the same group, they should be located in adjacent columns of \code{x}
#' and their corresponding index in \code{group.index} should be the same.
#' Denote the first group as \code{1}, the second \code{2}, etc.
#' If you do not fit a model with a group structure,
#' please set \code{group.index = NULL}. Default is \code{NULL}.
#' @return A list with class attribute 'bsrr' and named components:
#' \item{beta}{The best fitting coefficients.}
#'  \item{coef0}{The best fitting
#' intercept.}
#' \item{bestmodel}{The best fitted model for \code{type = "bss"}, the class of which is \code{"lm"}, \code{"glm"} or \code{"coxph"}.}
#' \item{loss}{The training loss of the fitting model.}
#' \item{s}{The model size.}
#' \item{lambda}{The shrinkage parameter.}
#' \item{family}{Type of the model.}
#' \item{nsample}{The sample size.}
#' \item{type}{Either \code{"bss"} or \code{"bsrr"}.}
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bsrr}}, \code{\link{summary.bsrr}}
#' \code{\link{coef.bsrr}}, \code{\link{predict.bsrr}}.
#' @references Wen, C., Zhang, A., Quan, S. and Wang, X. (2020). BeSS: An R
#' Package for Best Subset Selection in Linear, Logistic and Cox Proportional
#' Hazards Models, \emph{Journal of Statistical Software}, Vol. 94(4).
#' doi:10.18637/jss.v094.i04.
#' @examples
#'
#' #-------------------linear model----------------------#
#' # Generate simulated data
#' n <- 200
#' p <- 20
#' k <- 5
#' rho <- 0.4
#' seed <- 10
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lm.bsrr <- bsrr.one(x, y, s = 5, lambda = 0.01)
#' coef(lm.bsrr)
#' print(lm.bsrr)
#' summary(lm.bsrr)
#' pred.bsrr <- predict(lm.bsrr, newx = x_new)
#'
#' #-------------------logistic model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "binomial", beta = Tbeta, seed = seed)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' logi.bsrr <- bsrr.one(x, y, family = "binomial", s = 5, lambda = 0.01)
#' coef(logi.bsrr)
#' print(logi.bsrr)
#' summary(logi.bsrr)
#' pred.bsrr <- predict(logi.bsrr, newx = x_new)
#'
#'#-------------------poisson model----------------------#
#' Data <- gen.data(n, p, k, rho=0.3, family = "poisson", beta = Tbeta, seed = seed)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' poi.bsrr <- bsrr.one(x, y, family = "poisson", s = 5, lambda = 0.01)
#' coef(poi.bsrr)
#' print(poi.bsrr)
#' summary(poi.bsrr)
#' pred.bsrr <- predict(poi.bsrr, newx = x_new)
#'
#' #-------------------coxph model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "cox", beta = Tbeta, scal = 10)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140, ]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200, ]
#' cox.bsrr <- bsrr.one(x, y, family = "cox", s = 5, lambda = 0.01)
#' coef(cox.bsrr)
#' print(cox.bsrr)
#' summary(cox.bsrr)
#' pred.bsrr <- predict(cox.bsrr, newx = x_new)
#'#----------------------High dimensional linear models--------------------#
#'\dontrun{
#' data <- gen.data(n, p = 1000, k, family = "gaussian", seed = seed)
#'
#'# Best subset selection with SIS screening
#' fit <- bsrr.one(data$x, data$y, screening.num = 100, s = 5)
#'}
#'
#'#-------------------group selection----------------------#
#'beta <- rep(c(rep(1,2),rep(0,3)), 4)
#'Data <- gen.data(200, 20, 5, rho=0.4, beta = beta, seed =10)
#'x <- Data$x
#'y <- Data$y
#'
#'group.index <- c(rep(1, 2), rep(2, 3), rep(3, 2), rep(4, 3),
#'                 rep(5, 2), rep(6, 3), rep(7, 2), rep(8, 3))
#'lm.groupbsrr <- bsrr.one(x, y, s = 5, lambda = 0.01, group.index = group.index)
#'coef(lm.groupbsrr)
#'print(lm.groupbsrr)
#'summary(lm.groupbsrr)
#'pred.groupl0l2 <- predict(lm.groupbsrr, newx = x_new)
#'#-------------------include specified variables----------------------#
#'Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#'lm.bsrr <- bsrr.one(Data$x, Data$y, s = 5, always.include = 2)
#'
#' @export


bsrr.one <- function(x, y, family = c("gaussian", "binomial", "poisson", "cox"),
                             s, lambda= 0, always.include = NULL,
                             screening.num = NULL,
                             normalize = NULL, weight = NULL,
                             max.iter = 20,
                             group.index =NULL){
  if(length(s)>1) stop("bsrr.one needs only a single value for s.")
  if(length(lambda) > 1) stop("bsrr.one needs only a single value for lambda.")
  family <- match.arg(family)
  type <- 'bsrr'

  res <- bsrr(x, y, family = family,
              method ="sequential",
              tune = "gic",
              s.list=s, lambda.list = lambda,
              s.min=s, s.max=s,
              lambda.min = lambda, lambda.max = lambda, nlambda = 1,
              always.include = always.include,
              screening.num = screening.num,
              normalize = normalize, weight = weight,
              max.iter = max.iter, warm.start = TRUE,
              nfolds = 5,
              group.index =group.index,
              seed=NULL)
  res$s <- s
  res$bsrr.one <- TRUE
  res$call <- match.call()
  res$beta.all <- NULL
  res$lambda.list <- NULL
  res$s.list <- NULL
  if(type == 'bsrr'){
    res$ic.all <- NULL
    res$s.list <- NULL
    res$loss.all <- NULL
    res$beta.all <- NULL
    res$coef0.all <- NULL
    res$lambda.list <- NULL
    res$algorithm_type <- "L0L2"
    res$method <- NULL
    res$line.search <- NULL
    res$ic.type <- NULL
    res$lambda.max <- NULL
    res$lambda.min <- NULL
    if(!is.null(res$lambda.all)){
     res$lambda.all <- NULL
    }
    res$s.max <- NULL
    res$s.min <- NULL
    res$nlambda <-  NULL
  }else{
    res$ic.all <- NULL
    res$loss.all <- NULL
    res$beta.all <- NULL
    res$coef0.all <- NULL
    res$s.list <- NULL
    res$algorithm_type <- "PDAS"
    res$type <- type
    res$ic.type <- NULL
    res$s.max <- NULL
    res$s.min <- NULL
  }
  return(res)
}
