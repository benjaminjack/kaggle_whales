library(tidyverse)
library(keras)

source("config.R")


# Define the model -----------------------------------------------------------------------
siamese_net <- function() {
  
  input_shape <- c(FLAGS$width, FLAGS$height, 3)

  base_model <- application_vgg16(weights = "imagenet",
                                     include_top = FALSE,
                                     input_shape = input_shape)
  
  freeze_weights(base_model)
  
  siamese_branch <- keras_model_sequential(name = "siamese_branch") %>%
    base_model() %>%
    layer_flatten() %>% 
    layer_dense(units = 1024, activation = "sigmoid")
  
  summary(siamese_branch)
  
  left_input <- layer_input(shape = input_shape)
  left_features <- left_input %>% siamese_branch()
  
  right_input <- layer_input(shape = input_shape)
  right_features <- right_input %>% siamese_branch()
  
  output <- layer_lambda(list(left_features, right_features), function(x) k_abs(x[[1]] - x[[2]])) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model <- keras_model(list(left_input, right_input), output)

  model
}


