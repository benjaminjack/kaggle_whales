library(tidyverse)
library(keras)

source("config.R")

DIMS = 250

# Define the model -----------------------------------------------------------------------

base_model <- application_vgg16(weights = "imagenet", 
                                include_top = FALSE, 
                                input_shape = c(DIMS, DIMS, 3))

freeze_weights(base_model)

# L2 regularization seems to strike the best balance of training speed vs regularizer
siamese_branch <- keras_model_sequential() %>%
  base_model() %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu", kernel_regularizer = regularizer_l2(l = 1e-3))

summary(siamese_branch)

left_input <- layer_input(shape = c(DIMS, DIMS, 3))
left_features <- left_input %>% siamese_branch()

right_input <- layer_input(shape = c(DIMS, DIMS, 3))
right_features <- right_input %>% siamese_branch()

output <- layer_lambda(list(left_features, right_features), function(x) k_abs(x[[1]] - x[[2]])) %>%
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(list(left_input, right_input), output)

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

summary(model)

# Data loaders ----------------------------------------------------------------------------

pair_generator <- function(batch_size, size, labels, same_pair_labels) {
  left_input <- array(0, dim = c(batch_size, size, size, 3))
  right_input <- array(0, dim = c(batch_size, size, size, 3))
  targets <- array(0, dim = c(batch_size))
  
  for (i in 1:batch_size) {
    if (i %% 2 == 0) {
      left_image <- same_pair_labels %>% sample_n(1)
      right_image <- same_pair_labels %>% 
        filter(left_image$Id[[1]] == Id, left_image$Image != Image) %>%
        sample_n(1)
      # cat("same: ", left_image$Id[[1]], right_image$Id[[1]], "\n")
      targets[i] <- 1
    } else {
      left_image <- train_labels %>% sample_n(1)
      if (left_image$Image[[1]] == "new_whale") {
        right_image <- labels %>% filter(left_image$Image[[1]] != Image) %>% sample_n(1)
      } else {
        right_image <- labels %>% filter(left_image$Id[[1]] != Id) %>% sample_n(1)
      }
      # cat("different: ", left_image$Id[[1]], right_image$Id[[1]], "\n")
      targets[i] <- 0 
    }
    left_input[i,,,] <- image_load(file.path(FLAGS$data_dir, "train", left_image$Image[[1]]),
                                target_size = c(size, size)) %>% image_to_array() /255
    right_input[i,,,] <- image_load(file.path(FLAGS$data_dir, "train", right_image$Image[[1]]),
                                 target_size = c(size, size)) %>% image_to_array() / 255
  }
  list(list(left_input, right_input), targets)
}

train_labels <- read_csv("train_labels.csv")
train_same_pair_labels <- train_labels %>%
  group_by(Id) %>%
  filter(Id != "new_whale", n() >= 2) %>%
  ungroup()

train_pair_generator <- function(batch_size = 4, size = DIMS) {
  pair_generator(batch_size = batch_size, 
                 size = size, 
                 labels = train_labels,
                 same_pair_labels = train_same_pair_labels)
}

val_labels <- read_csv("val_labels.csv")
val_same_pair_labels <- val_labels %>%
  group_by(Id) %>%
  filter(Id != "new_whale", n() >= 2) %>%
  ungroup()

val_pair_generator <- function(batch_size = 4, size = DIMS) {
  pair_generator(batch_size = batch_size, 
                 size = size, 
                 labels = val_labels,
                 same_pair_labels = val_same_pair_labels)
}

# Fit model --------------------------------------------------------------------------

callbacks_list <- list(
  callback_model_checkpoint(
    filepath = "whales.h5",
    monitor = "val_loss",
    save_best_only = TRUE
  )
)

history <- model %>% fit_generator(
  generator = train_pair_generator,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_pair_generator,
  validation_steps = 100,
  callbacks = callbacks_list
)


