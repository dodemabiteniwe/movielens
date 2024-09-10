##########################################################
# Exploratory analysis
##########################################################
library(tidyverse)
library(caret)
library(Matrix)
library(grDevices)


load("rdas/edx.rda")
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01")
dim(edx)

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
      xaxt = 'n', yaxt = 'n')+ abline(h = 0:100 + 0.5, v = 0:100 + 0.5, col = "grey")# Hide axis ticks
axis(1, at = seq(0,101,10), labels = TRUE)
axis(2, at = seq(0,101,10), labels = TRUE)

################################################################################
#Distributions of variables
#########################################

# Load the necessary library
library(dplyr)
library(ggplot2)
library(cowplot)
#library(data.table)
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
  scale_x_continuous(n.breaks = 10) 
  #theme_economist()

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
  scale_x_continuous(n.breaks = 10) +
  theme_economist() 
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
  scale_x_log10(n.breaks = 10) +
  theme_economist()

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
  scale_x_continuous(n.breaks = 10) +
  theme_economist()
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
  scale_x_log10(n.breaks = 10) +
  theme_economist()

ggarrange(plot_user1, plot_user2,  
          labels = c("A", "B"),
          ncol = 2, nrow = 1)



#######################################################################################
# System modelling
####################################################################################
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

###############################################"
# Model_A: Linear Regression Model
############################"

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

y_hat_regularized <- final_holdout_test %>% 
  left_join(b_i_regularized, by = "movieId") %>%
  left_join(b_u_regularized, by = "userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

# Evaluation 

evaluation <- tibble(Model = c("Cinematch", "The Netflix Prize", "Linear Regression Model"),
                     MAE = c(NA, NA, Metrics::mae(final_holdout_test$rating, y_hat_regularized)),
                     MSE = c(NA, NA, Metrics::mse(final_holdout_test$rating, y_hat_regularized)),
                     RMSE = c(0.9525, 0.85725, Metrics::rmse(final_holdout_test$rating, y_hat_regularized)))
print(evaluation)

###############################################"
# Model_B: Matrix factorization
############################

##### Data set reorganisation in package Recosytem format################

set.seed(1)
train_rec <- with(train_set, data_memory(user_index = userId, 
                                                item_index = movieId,
                                                rating     = rating))
test_rec <- with(test_set, data_memory(user_index = userId, 
                                              item_index = movieId, 
                                              rating     = rating))
valid_rec <- with(final_holdout_test, data_memory(user_index = userId, 
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
                                                       niter = 20))

#####Prediction##############

#y_hat_MatFac <-  recom_system$predict(test_rec, out_memory())
y_hat_MatFac <-  recom_system$predict(valid_rec, out_memory())

#############Evaluation ###############

evaluation <- bind_rows(evaluation,
                        tibble(Model = "Matrix factorization",
                               MAE  = Metrics::mae(final_holdout_test$rating, y_hat_MatFac),
                               MSE  = Metrics::mse(final_holdout_test$rating, y_hat_MatFac),
                               RMSE = Metrics::rmse(final_holdout_test$rating, y_hat_MatFac)))
print(evaluation)

###############################################"
# Model_C: Others algorithms with recommendalab
############################
head(edx)

library("recommenderlab")

# Convert the data to a realRatingMatrix
rat_mat <- as(select(edx,userId,movieId,rating), "realRatingMatrix")
test_mat <- as(select(final_holdout_test,userId,movieId,rating), "realRatingMatrix")
#Coertion to convert data into realRatingMatrix object and normalization

#r <- as(m, "realRatingMatrix")
r_m <- normalize(rat_mat)
rtest_m <- normalize(test_mat)
#Small portions of rating matrices can be visually inspected using image()."
#set.seed(1234)
#r_s <- sample(rat_mat, 60)
#r_sm <- sample(r_m, 60)
#image(r_s, main = "Raw Ratings")
#image(r_sm, main = "Normalized Ratings")
hist(getRatings(r_m), breaks=100)

###############################
# Build and Evaluate Models
######################

set.seed(123)
sample_split <- sample(x = 1:nrow(rat_mat), size = 0.8 * nrow(rat_mat))
train_data <- rat_mat[sample_split, ]
test_data <- rat_mat[-sample_split, ]




# Define the models to be evaluated
model_list <- list(
  "UBCF" = list(name = "UBCF", param = list(nn = 30)),
  "IBCF" = list(name = "IBCF", param = list(k = 30)),
  "SVD" = list(name = "SVD", param = list(k = 50))
)

# Function to evaluate models
evaluate_models <- function(train, test, models) {
  results <- list()
  
  for (model_name in names(models)) {
    cat("Evaluating model:", model_name, "\n")
    
    model <- Recommender(train, method = models[[model_name]]$name, parameter = models[[model_name]]$param)
    prediction <- predict(model, test, type = "ratings")
    
    rmse <- calcPredictionAccuracy(prediction, test, byUser = FALSE)$RMSE
    mse <- calcPredictionAccuracy(prediction, test, byUser = FALSE)$MSE
    mae <- calcPredictionAccuracy(prediction, test, byUser = FALSE)$MAE
    
    results[[model_name]] <- list(RMSE = rmse, MSE = mse, MAE = mae)
  }
  
  return(results)
}

# Run the evaluation
results <- evaluate_models(train_data, test_data, model_list)
results

model <- Recommender(train_data, method = "UBCF")
prediction <- predict(model, test_data, type = "ratings")

