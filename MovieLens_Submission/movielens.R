##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
#set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
save(edx, file = "rdas/edx.rda")
save(final_holdout_test, file = "rdas/final_holdout_test.rda")




#------------------------------ANALYSIS------------------------------------------------------------------------------------



##########################################################
# Exploratory analysis
##########################################################


# Load the necessary library and dataset
library(tidyverse)
library(caret)
library(Matrix)
library(grDevices)
library(dplyr)
library(ggplot2)
library(cowplot)
library(ggthemes)
library(lubridate)
library(Metrics)
library(recosystem)
library(scales)
library(stringr)
library(tibble)
library(tidyr)
if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/ggpubr")
library(ggpubr)
library("recommenderlab")
library(knitr)
library(kableExtra)

load("../rdas/edx.rda")
load("../rdas/final_holdout_test.rda")
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01")
final_holdout_test$timestamp <- as.POSIXct(final_holdout_test$timestamp, origin = "1970-01-01")

# First lines of edx data
kable(
  head(edx,5), booktabs = TRUE, caption = "First lines of edx data.")%>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))
# First lines of validation data
kable(
  head(final_holdout_test,5), booktabs = TRUE, caption = "First lines of validation data.")%>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))


#####################################################################
# Sparsity of MovieLens Ratings Matrix
################################################

# Sample 100 random users and movies
set.seed(1)
sampled_users <- sample(unique(edx$userId), 100)
sampled_movies <- sample(unique(edx$movieId), 100)
sampled_data <- subset(edx, userId %in% sampled_users  & movieId %in% sampled_movies)
sparse_matrix <- sparseMatrix(i = as.integer(factor(sampled_data$userId, levels = sampled_users)),
                              j = as.integer(factor(sampled_data$movieId, levels = sampled_movies)),
                              x = sampled_data$rating)
full_matrix <- as.matrix(sparse_matrix)
full_matrix[full_matrix == 0] <- NA
# Plot the matrix using image
image(x = seq_len(ncol(full_matrix)), y = seq_len(nrow(full_matrix)), z = t(full_matrix[nrow(full_matrix):1, ]),
      main = "Sparsity of MovieLens Ratings Matrix",
      xlab = "Users", ylab = "Movies",
      col = "yellow", breaks = c(min(full_matrix, na.rm = TRUE), max(full_matrix, na.rm = TRUE)),
      axes = TRUE, # Enable axes for clarity
      xaxt = 'n', yaxt = 'n')
axis(1, at = seq(0,101,10), labels = TRUE)
axis(2, at = seq(0,101,10), labels = TRUE)


################################################################################
#Distributions of variables
#########################################

# Rating distribution
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_bar(stat = "identity", fill = "#db88ff") +
  ggtitle("Rating Distribution") +
  xlab("Rating") +
  ylab("Count") +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(n.breaks = 10) +
  theme_economist()

######## Rating per movies ########
plot_mov1 <- edx%>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = movieId, y = count)) +
  geom_point(alpha = 0.2, color = "#4020dd") +
  geom_smooth(color = "red") +
  ggtitle("Ratings per movie") +
  xlab("Movies") +
  ylab("Number of ratings") +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(n.breaks = 10)
########### Movies' rating histogram ###################
plot_mov2 <- edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = count)) +
  geom_histogram(fill = "#db88ff", color = "#4020dd") +
  ggtitle("Movies' rating histogram") +
  xlab("Rating count") +
  ylab("Number of movies") +
  scale_y_continuous(labels = comma) +
  scale_x_log10(n.breaks = 10)

ggarrange(plot_mov1, plot_mov2,  
          labels = c("A", "B"),
          ncol = 2, nrow = 1)

######## Rating per users ########
plot_user1 <- edx%>%
  group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = userId, y = count)) +
  geom_point(alpha = 0.2, color = "#4020dd") +
  geom_smooth(color = "red") +
  ggtitle("Ratings per user") +
  xlab("Users") +
  ylab("Number of ratings") +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(n.breaks = 10)
########### Users' rating histogram ###################
plot_user2 <- edx %>%
  group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = count)) +
  geom_histogram(fill = "#db88ff", color = "#4020dd") +
  ggtitle("Users' rating histogram") +
  xlab("Rating count") +
  ylab("Number of users") +
  scale_y_continuous(labels = comma) +
  scale_x_log10(n.breaks = 10)

ggarrange(plot_user1, plot_user2,  
          labels = c("A", "B"),
          ncol = 2, nrow = 1)



#######################################################################################
# System modelling
####################################################################################

###############################################"
# Modeling approach A: Popular, random, svd algorithms with recommendalab
############################

# Convert the data to a realRatingMatrix
rat_mat <- as(select(edx,userId,movieId,rating), "realRatingMatrix")


###############################
# Build and Evaluate Models
######################

set.seed(123)
sample_split <- sample(x = 1:nrow(rat_mat), size = 0.8 * nrow(rat_mat))
train_data <- rat_mat[sample_split, ]
test_data <- rat_mat[-sample_split, ]

# Define the models to be evaluated
model_list <- c("POPULAR","RANDOM", "SVD")

# Function to evaluate models
evaluate_models <- function(train, test, models) {
  results <- list()
  
  for (model_name in models) {
    model <- Recommender(train, method = model_name)
    prediction <- predict(model, test, type = "ratingMatrix")
    
    results[[model_name]] <- calcPredictionAccuracy(prediction, test, byUser = FALSE)
  }
  
  return(results)
}

# Run the evaluation
results <- evaluate_models(train_data, test_data, model_list)

evaluation <- tibble(Model = c("RANDOM", "POPULAR", "SVD"),
                     MAE = c(results$RANDOM[[3]], results$POPULAR[[3]], results$SVD[[3]]),
                     MSE = c(results$RANDOM[[2]], results$POPULAR[[2]], results$SVD[[2]]),
                     RMSE = c(results$RANDOM[[1]], results$POPULAR[[1]], results$SVD[[1]]))

kable(
  evaluation, booktabs = TRUE, caption = "Algorithms performance.")%>%
  kable_styling(latex_options = "HOLD_position")

###############################################"
# Modeling approach B: Linear Regression Model
############################
# Train-test split
###################################
set.seed(1)
train_index <- createDataPartition(edx$rating, times = 1, p = 0.9, list = FALSE)
train_set <- edx[train_index,]
temp_test_set <- edx[-train_index,]

tibble(Dataset = c("edx_data", "train_set", "temp_test_set"),
       "Number of ratings" = c(nrow(edx), nrow(train_set), nrow(temp_test_set)))
# Make sure userId and movieId in the testing set are also in the training set set
test_set <- temp_test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
# Add rows removed from the testing set back into the training set set
removed <- anti_join(temp_test_set, test_set)
train_set <- rbind(train_set, removed)

# Lambda's parameter for regularisation in model_A

# Regularization function
regularization <- function(lambda, train_set, test_set){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(Metrics::rmse(predicted_ratings, test_set$rating))
}

# Lambda
lambdas <- seq(0, 10, 0.25)
lambdas_rmse <- sapply(lambdas,
                       regularization, 
                       train_set = train_set, 
                       test_set = test_set)
lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)
print(lambdas_tibble)

lambdas_tibble %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  ggtitle("Lambda's effect on RMSE") +
  xlab("Lambda") +
  ylab("RMSE") +
  scale_y_continuous(n.breaks = 6, labels = comma) +
  scale_x_continuous(n.breaks = 10) +
  theme_economist() +
  theme(axis.title.x = element_text(vjust = -5, face = "bold"), 
        axis.title.y = element_text(vjust = 10, face = "bold"), 
        plot.margin = margin(0.7, 0.5, 1, 1.2, "cm"))

# Regularized linear model construction

lambda <- lambdas[which.min(lambdas_rmse)]

mu <- mean(train_set$rating)

b_i_regularized <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_regularized <- train_set %>% 
  left_join(b_i_regularized, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

y_hat_regularized <- test_set %>% 
  left_join(b_i_regularized, by = "movieId") %>%
  left_join(b_u_regularized, by = "userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

# Evaluation 

evaluation <-bind_rows(evaluation,
                       tibble(Model = "Linear Regression Model",
                              MAE  = Metrics::mae(test_set$rating, y_hat_regularized),
                              MSE  = Metrics::mse(test_set$rating, y_hat_regularized),
                              RMSE = Metrics::rmse(test_set$rating, y_hat_regularized)))

kable(
  evaluation, booktabs = TRUE, caption = "Algorithms performance.")%>%
  kable_styling(latex_options = "HOLD_position")

###############################################"
# Modeling approach C: Matrix factorization
############################

##### Data set reorganisation in package Recosytem format################

set.seed(1)
train_rec <- with(train_set, data_memory(user_index = userId, 
                                         item_index = movieId,
                                         rating     = rating))
test_rec <- with(test_set, data_memory(user_index = userId, 
                                       item_index = movieId, 
                                       rating     = rating))

##### Model objet#######

recom_system <- Reco()

#### Model is tuned ######
tun <- recom_system$tune(train_rec, opts = list(dim = c(10, 20, 30),
                                                lrate = c(0.1, 0.2),
                                                nthread  = 4,
                                                niter = 10))


##### Model training #############

recom_system$train(train_rec, opts = c(tun$min,
                                       nthread = 4,
                                       niter = 31))

#####Prediction##############
y_hat_MatFac <-  recom_system$predict(test_rec, out_memory())

#############Evaluation ###############
evaluation <- bind_rows(evaluation,
                        tibble(Model = "Matrix factorization",
                               MAE  = Metrics::mae(test_set$rating, y_hat_MatFac),
                               MSE  = Metrics::mse(test_set$rating, y_hat_MatFac),
                               RMSE = Metrics::rmse(test_set$rating, y_hat_MatFac)))
kable(
  evaluation, booktabs = TRUE, caption = "Algorithms performance on test data.")%>%
  kable_styling(latex_options = "HOLD_position")

#######################################################################
#Model performance on the final holdout data
######################################################
set.seed(1)
train_rec2 <- with(edx, data_memory(user_index = userId, 
                                    item_index = movieId,
                                    rating     = rating))
valid_rec <- with(final_holdout_test, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating     = rating))
#### Model is tuned ######
tun2 <- recom_system$tune(train_rec2, opts = list(dim = c(10, 20, 30),
                                                  lrate = c(0.1, 0.2),
                                                  nthread  = 4,
                                                  niter = 10))

##### Model training #############
recom_system$train(train_rec2, opts = c(tun2$min,
                                        nthread = 4,
                                        niter = 31))

#####Prediction##############
y_hat_MatFac2 <-  recom_system$predict(valid_rec, out_memory())
Metrics::rmse(final_holdout_test$rating, y_hat_MatFac2)

##############################################################
# Top 10 movie recommendation by the final model
########################################################
top10_pred <- tibble(Title = final_holdout_test$title, y_hat = y_hat_MatFac2) %>%
  arrange(desc(y_hat)) %>%
  select(Title) %>%
  unique() %>%
  slice_head(n = 10)
top10_pred_df <- data.frame(Title = top10_pred,
                            Rating = rep(NA, 10),
                            Count = rep(NA, 10))

for (i in 1:10) {
  ind <- which(final_holdout_test$title == as.character(top10_pred[i,]))
  top10_pred_df$Rating[i] <- mean(final_holdout_test$rating[ind])
  top10_pred_df$Count[i] <- sum(
    final_holdout_test$title == as.character(top10_pred[i,])
  )
}
kable(
  top10_pred_df, booktabs = TRUE, caption = "Top 10 movie recommendation by the final model.")%>%
  kable_styling(latex_options = "HOLD_position")

#-------------------------------------End.----------------------------------------------------------------------------------











