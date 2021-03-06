% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions.R
\name{standardize.columns}
\alias{standardize.columns}
\title{Standardize columns}
\usage{
standardize.columns(data, variables)
}
\arguments{
\item{data}{data.frame with variables of interest}

\item{variables}{vector of variables to process}
}
\value{
The same data frame with variables replaced with their z-scores
}
\description{
Calculates z-score for one or several variables
}
\examples{
standardize.columns(df, c('var1','var2','var3'))

}

