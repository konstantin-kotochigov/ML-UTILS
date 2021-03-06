% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions.R
\name{get.var.importance}
\alias{get.var.importance}
\title{Calculate combined importance measure of each attribute}
\usage{
get.var.importance(x, y)
}
\arguments{
\item{x}{data frame with variables of interest}

\item{y}{target variable}
}
\value{
data frame with variable names sorted by combined importance statistics
}
\description{
Creates a data frame with calculated chi-squared, AUC and RandomForest measures combined into one score and sorted by it
}

