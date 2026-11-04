install.packages("tidyverse")
library(tidyverse)
data <- read.csv("Solar Power Plant Data.csv")
names(data)
head(data)
library(caret)
library(rpart)
library(rpart.plot)
library(Metrics)
library(ggplot2)

# We'll work with a clean copy
energy_df <- data %>%
  select(
    WindSpeed,
    Sunshine,
    AirPressure,
    Radiation,
    AirTemperature,
    RelativeAirHumidity,
    SystemProduction
  )

# See summary stats
summary(energy_df)

# Drop rows with any missing values (simple & safe for now)
energy_df <- na.omit(energy_df)

# Check final size
dim(energy_df)

# Train/Test Split (80% train, 20% test)
set.seed(123)  # stay consistent every run
library(caret)

train_index <- createDataPartition(
  energy_df$SystemProduction,
  p = 0.8,
  list = FALSE
)

train_data <- energy_df[train_index, ]
test_data  <- energy_df[-train_index, ]

dim(train_data)
dim(test_data)
# Train a Decision Tree Regression model
tree_model <- rpart(
  SystemProduction ~ WindSpeed + Sunshine + AirPressure + Radiation +
    AirTemperature + RelativeAirHumidity,
  data = train_data,
  method = "anova",   # regression tree, not classification
  control = rpart.control(
    cp = 0.01,        # complexity parameter (smaller = deeper tree)
    minsplit = 20,    # min rows needed to try a split
    maxdepth = 10     # safety limit so it doesn't get insanely deep
  )
)

# See model summary in console
print(tree_model)

# Visualize the decision tree
rpart.plot(
  tree_model,
  type = 2,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Decision Tree for Solar Energy Output Prediction"
)
# ---- STEP 6: Model Evaluation ----

# Predict energy output on test data
test_predictions <- predict(tree_model, newdata = test_data)

# Actual values
actuals <- test_data$SystemProduction

# Compute evaluation metrics
mae_value  <- mae(actuals, test_predictions)
rmse_value <- rmse(actuals, test_predictions)

# R-squared manually
ss_res <- sum((actuals - test_predictions)^2)
ss_tot <- sum((actuals - mean(actuals))^2)
r_squared <- 1 - (ss_res / ss_tot)

# Print the metrics
cat("Model Performance Metrics:\n")
cat("MAE  :", mae_value, "\n")
cat("RMSE :", rmse_value, "\n")
cat("R²   :", r_squared, "\n")

# ---- Visualization ----
results_df <- data.frame(
  Actual = actuals,
  Predicted = test_predictions
)

ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Solar Energy Output",
    x = "Actual Power Output",
    y = "Predicted Power Output"
  ) +
  theme_minimal()
# ---- STEP 7: Feature Importance ----

# Extract importance values from the tree model
importance_values <- tree_model$variable.importance

# Convert to a data frame for plotting
importance_df <- data.frame(
  Feature = names(importance_values),
  Importance = as.numeric(importance_values)
)

# Sort by importance descending
importance_df <- importance_df %>%
  arrange(desc(Importance))

# Print the table in console
print(importance_df)

# Plot feature importance
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Feature Importance in Solar Power Prediction",
    x = "Feature",
    y = "Importance Score"
  ) +
  theme_minimal()
