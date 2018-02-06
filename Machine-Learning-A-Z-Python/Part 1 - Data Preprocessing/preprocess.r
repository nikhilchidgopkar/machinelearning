setwd("C:/Users/nikhilc/Google Drive/_Data Science/Udmey_A2Z_ML/Machine Learning A-Z - Self/Part 1 - Data Preprocessing");
dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary,FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)


dataset$Country = factor(dataset$Country,
                         level = c('France','Spain', 'Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         level = c('Yes','No'),
                         labels = c(1,0))

# Splitting Test & Training set
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)

train_set = subset(dataset,split==TRUE)
test_set =subset(dataset,split==FALSE)




# Feature Scaling

train_set[,2:3] = scale(train_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])