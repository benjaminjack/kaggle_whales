library(tidyverse)
library(keras)

source("config.R")

DIMS <- FLAGS$dims

# Define the model -----------------------------------------------------------------------
siamese_net <- function() {
  base_model <- application_vgg16(weights = "imagenet", 
                                  include_top = FALSE, 
                                  input_shape = c(DIMS, DIMS, 3))
  
  freeze_weights(base_model)
  
  # L2 regularization seems to strike the best balance of training speed vs regularizer
  siamese_branch <- keras_model_sequential(name = "siamese_branch") %>%
    base_model() %>%
    layer_flatten() %>%
    layer_dense(units = 2048, activation = "sigmoid")
  
  left_input <- layer_input(shape = c(DIMS, DIMS, 3))
  left_features <- left_input %>% siamese_branch()
  
  right_input <- layer_input(shape = c(DIMS, DIMS, 3))
  right_features <- right_input %>% siamese_branch()
  
  output <- layer_lambda(list(left_features, right_features), function(x) k_abs(x[[1]] - x[[2]])) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model <- keras_model(list(left_input, right_input), output)

  model %>% compile(
    optimizer = optimizer_rmsprop(lr = 1e-5),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
}


