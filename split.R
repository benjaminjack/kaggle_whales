library(tidyverse)
library(purrr)

source("config.R")

print(FLAGS)

labels <- read_csv(paste0(FLAGS$data_dir, "train.csv"))

val_labels <- labels %>% sample_frac(0.2)
train_labels <- labels %>% anti_join(val_labels, by = c("Image", "Id"))

write_csv(val_labels, "val_labels.csv")
write_csv(train_labels, "train_labels.csv")
