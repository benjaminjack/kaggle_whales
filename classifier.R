
# Build kNN classifier ---------------------------------------------------------------------------

classifier <- keras_model_sequential %>% 
  siamese_branch() %>%
  layer_dense(units = 16, activation = relu) %>%
  layer_dense(units = 1, activation = "softmax")

freeze_weights(siamese_branch)

classifier %>% compile()