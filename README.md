# Practical_Machine_Learning- Project write up
In this project write-up, I have used the data from Human Activity Recognition (HAR). The aim was to  model  Class variable based on the data of various sensor values.

After loading the data in R environment and doing some basic univariate analysis, I have realized that some columns have a lot of missing (NA) values. 
I have decided to remove those variables from the data set as the treatement of those missing values are quite difficult with not having enough information on each variables. 


setwd("D:/Coursera/machine_learning")
library(lattice)
library(ggplot2)
library(caret)
# Load the training data set
trainingAll <- read.csv("pml-training.csv",na.strings=c("NA",""))
# Discard columns with NAs
NAs <- apply(trainingAll, 2, function(x) { sum(is.na(x)) })
trainingValid <- trainingAll[, which(NAs == 0)]


This resulted in 60 columns (variables), instead of 160.


After having removed the columns with missing values, I have proceeded to create a subset of the training data set to reduce the computation time as i am working on desktop version of R studio. working on the entire data set contained 19622 rows (observations) would be computationally quite expensive.


# Create a subset of trainingValid data set
trainIndex <- createDataPartition(y = trainingValid$classe, p=0.5,list=FALSE)
trainData <- trainingValid[trainIndex,]


 Therefore I have decided to take 50% of the whole HAR data set as a representative sample.


Moreover, after creating this subset, I also removed the columns related to timestamps, the X column, user_name, and new_window because they were not sensor values, so I thought they would not help much (or at all) for prediction:


# Remove useless predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]

As a result, I had a subset of HAR data set that had only ##### rows of of 54 variables.


Then, based on the suggestion of the instructor (“… how you used cross validation”), I've used k-fold cross validation with K=4 
 After setting the trainControl, I have finally used the Random Forests (rf) algorithm in the following manner:
 
 
 # Configure the train control for cross-validation
tc = trainControl(method = "cv", number = 4)

library(randomForest)
# Fit the model using Random Forests algorithm
modFit <- train(trainData$classe ~.,
                data = trainData,
                method="rf",
                trControl = tc,
                prox = TRUE,
                allowParallel = TRUE)
                
 I expected relatively good model performance, and a relatively low out of sample error rate:
 
 print(modFit)
print(modFit$finalModel)





# Load test data
testingAll = read.csv("pml-testing.csv",na.strings=c("NA",""))

# Only take the columns of testingAll that are also in trainData
testing <- testingAll[ , which(names(testingAll) %in% names(trainData))]

# Run the prediction
pred <- predict(modFit, newdata = testing)

# Utility function provided by the instructor
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)







