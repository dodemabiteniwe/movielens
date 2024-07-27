##########################################################
# Exploratory analysis
##########################################################
library(tidyverse)
library(caret)
load("rdas/edx.rda")
str(edx)
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01")
# number of rows and columns of data
dim(edx)
# number of different movies and users in the edx dataset
n_distinct(edx$movieId)
n_distinct(edx$userId)
n_distinct(edx$genres)
