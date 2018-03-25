library(tensorflow)

FLAGS <- flags(
  flag_string("data_dir", "~/Data/kaggle/whale-categorization-playground/"),
  flag_integer("width", 224),
  flag_integer("height", 224),
  flag_integer("batch_size", 5)
)