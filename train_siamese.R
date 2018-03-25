library(tidyverse)
library(keras)

source("config.R")
source("model.R")

# Data loaders ----------------------------------------------------------------------------

train_image_gen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 30,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2
)

test_image_gen <- image_data_generator(rescale = 1/255)

pair_generator <- function(batch_size, width, height, labels, same_pair_labels, augment=FALSE) {
  left_input <- array(0, dim = c(batch_size, width, height, 3))
  right_input <- array(0, dim = c(batch_size, width, height, 3))
  targets <- array(0, dim = c(batch_size))
  
  for (i in 1:batch_size) {
    if (i %% 2 == 0) {
      # Get two images of the same ID
      left_image <- same_pair_labels %>% sample_n(1)
      right_image <- same_pair_labels %>% 
        filter(left_image$Id[[1]] == Id, left_image$Image != Image) %>%
        sample_n(1)
      targets[i] <- 1
    } else {
      # Get two images with different IDs
      left_image <- train_labels %>% sample_n(1)
      if (left_image$Image[[1]] == "new_whale") {
        right_image <- labels %>% filter(left_image$Image[[1]] != Image) %>% sample_n(1)
      } else {
        right_image <- labels %>% filter(left_image$Id[[1]] != Id) %>% sample_n(1)
      }
      targets[i] <- 0 
    }
    left_input[i,,,] <- image_load(file.path(FLAGS$data_dir, "train", left_image$Image[[1]]),
                                   target_size = c(width, height)) %>% image_to_array()
    right_input[i,,,] <- image_load(file.path(FLAGS$data_dir, "train", right_image$Image[[1]]),
                                    target_size = c(width, height)) %>% image_to_array()
  }
  if (augment == TRUE) {
    image_gen <- train_image_gen
  } else {
    image_gen <- test_image_gen
  }
  
  left_gen <- flow_images_from_data(left_input, generator = image_gen, batch_size = batch_size, shuffle = FALSE)
  right_gen <- flow_images_from_data(right_input, generator = image_gen, batch_size = batch_size, shuffle = FALSE)
  
  list(list(generator_next(left_gen), generator_next(right_gen)), targets)
}

train_labels <- read_csv("train_labels.csv")
train_same_pair_labels <- train_labels %>%
  group_by(Id) %>%
  filter(Id != "new_whale", n() >= 2) %>%
  ungroup()

train_pair_generator <- function() {
  pair_generator(batch_size = FLAGS$batch_size, 
                 width = FLAGS$width,
                 height = FLAGS$height,
                 labels = train_labels,
                 same_pair_labels = train_same_pair_labels,
                 augment = TRUE)
}

val_labels <- read_csv("val_labels.csv")
val_same_pair_labels <- val_labels %>%
  group_by(Id) %>%
  filter(Id != "new_whale", n() >= 2) %>%
  ungroup()

val_pair_generator <- function() {
  pair_generator(batch_size = FLAGS$batch_size, 
                 width = FLAGS$width,
                 height = FLAGS$height,
                 labels = val_labels,
                 same_pair_labels = val_same_pair_labels,
                 augment = FALSE)
}

# Fit model --------------------------------------------------------------------------

model <- siamese_net()

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

callbacks_list <- list(
  callback_model_checkpoint(
    filepath = "whales.h5",
    monitor = "val_loss",
    save_weights_only = TRUE,  # Keras doesn't like saving models with lambda layers
    save_best_only = TRUE
  )
)

model %>% fit_generator(
  generator = train_pair_generator,
  steps_per_epoch = 500,
  epochs = 100,
  validation_data = val_pair_generator,
  validation_steps = 100,
  callbacks = callbacks_list
)


# Fine tune ---------------------------------------------------------------------------

callbacks_list <- list(
  callback_model_checkpoint(
    filepath = "whales_fine_tune.h5",
    monitor = "val_loss",
    save_weights_only = TRUE,  # Keras doesn't like saving models with lambda layers
    save_best_only = TRUE
  )
)

load_model_weights_hdf5(model, "whales.h5")

unfreeze_weights(base_model, from = "block3_conv1")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  generator = train_pair_generator,
  steps_per_epoch = 500,
  epochs = 50,
  validation_data = val_pair_generator,
  validation_steps = 100,
  callbacks = callbacks_list
)