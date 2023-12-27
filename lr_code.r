%%R
# Read CSV file
data <- read.csv("drive/MyDrive/train.csv")

# create a new predictor variable
data$bidaskprice_percentagediff <- ((data$bid1 - data$ask1) / data$ask1) * 100

data$bidaskvol_percentagediff <- ((data$bid1vol - data$ask1vol) / data$ask1vol) * 100

data$top3bidaskvol_percentagediff <- ((data$bid1vol + data$bid2vol + data$bid3vol  - data$ask1vol -data$ask2vol-data$ask3vol ) / (data$ask1vol+data$ask2vol+data$ask3vol)) * 100
# Specify the target column
target_column <- "y"

# Specify predictor columns
predictor_columns <- c("bid1", "ask1",  "transacted_qty", "bid1vol", "ask1vol", "d_open_interest", "bidaskprice_percentagediff", "bidaskvol_percentagediff", "top3bidaskvol_percentagediff")  # you can add more columns  from the train.csv file here as features

# Create a formula for linear regression
formula <- as.formula(paste(target_column, "~", paste(predictor_columns, collapse = "+")))

# logistic regression
logistic_model <- glm(formula, data = data, family = "binomial")

# Print the summary of the model
summary(logistic_model)


# Install and load the glmnet
#install.packages("glmnet")
library(glmnet)

# Assuming you have already read the data and created the percentage_difference variable
# as shown in the previous responses

# Specify the target column
target_column <- "y"

# Specify predictor columns
predictor_columns <- c("bid1", "ask1",  "transacted_qty", "bid1vol", "ask1vol", "d_open_interest", "bidaskprice_percentagediff", "bidaskvol_percentagediff", "top3bidaskvol_percentagediff")  # you can add more columns  from the train.csv file here as features


# Create a matrix of predictors
X <- as.matrix(data[, predictor_columns])

# Create the response variable
y <- data[[target_column]]

# Fit logistic regression model with Lasso regularization
lasso_model <- cv.glmnet(X, y, family = "binomial", alpha = 1)

# Print the cross-validated selected lambda (penalty parameter)
print(lasso_model$lambda.min)

# Print the coefficients of the selected model
coef(lasso_model, s = "lambda.min")


# Fit logistic regression model with Ridge regularization
ridge_model <- cv.glmnet(X, y, family = "binomial", alpha = 0)

# Print the cross-validated selected lambda (penalty parameter)
print(ridge_model$lambda.min)

# Print the coefficients of the selected model
coef(ridge_model, s = "lambda.min")




# Install and load the glmnet
#install.packages("dplyr")

library(glmnet)
library(dplyr)


# Assuming you have already read the data and created the percentage_difference variable
# as shown in the previous responses
# Specify the target column
target_column <- "y"

# Specify predictor columns
predictor_columns <- c("bid1", "ask1",  "transacted_qty", "bid1vol", "ask1vol", "d_open_interest", "bidaskprice_percentagediff", "bidaskvol_percentagediff", "top3bidaskvol_percentagediff")  # you can add more columns  from the train.csv file here as features

# Create interaction terms by hard-coding pairs of variables
interaction_term_1 <- interaction(data$bid1, data$bid1vol) # bid1 * bid1vol
interaction_term_2 <- interaction(data$ask1, data$ask1vol) # ask1 * ask1vol

# Combine original predictors with interaction terms
data_with_interactions <- cbind(data[, predictor_columns], 
                                as.character(interaction_term_1),
                                as.character(interaction_term_2))

# Specify predictor columns, including the new variable and interaction terms
predictors_with_interactions <- c(predictor_columns, 
                                  as.character(interaction_term_1),
                                  as.character(interaction_term_2))

# Create a matrix of predictors
X <- model.matrix(~., data = data_with_interactions)[, -1]

# Create the response variable
y <- data[[target_column]]

# Fit logistic regression model with Lasso regularization
lasso_model_withinteraction <- cv.glmnet(X, y, family = "binomial", alpha = 1)

# Print the cross-validated selected lambda (penalty parameter)
print(lasso_model_withinteraction$lambda.min)

# Print the coefficients of the selected model
coef(lasso_model_withinteraction, s = "lambda.min")
