# Practical_Machine_Learning- Project write up
In this project write-up, I have used the data from Human Activity Recognition (HAR) as described by Jeff. The aim is to  model the   "Class" type based on the data of various sensor values.

After loading the data in R environment and doing some basic univariate analysis, I have realized that some columns have a lot of missing (NA) values. 
I have decided to remove those variables from the data set as the treatement of those missing values are quite difficult with not having enough understanding or the measurement unit on each variables. The training data has been loaded in R environment by executing the following code.


setwd("D:/Coursera/machine_learning") # to set up working directory
library(lattice)
library(ggplot2)
library(caret)
# Load the training data set
trainingAll <- read.csv("pml-training.csv",na.strings=c("NA",""))
# Remove columns with NAs
NAs <- apply(trainingAll, 2, function(x) { sum(is.na(x)) })
trainingValid <- trainingAll[, which(NAs == 0)]

This resulted in 60 predictor variables, instead of working with  160 variables .

After having removed the columns with missing values, I have proceeded to create a subset of the training data set to reduce the computation time as i am working on desktop version of R studio with limited capacity of RAM. Working on the entire data set contained 19,622 rows (observations) would be computationally quite expensive. Therefore I have decided to take 50% of the entire training data setto develop the random forest model. I decided to build random forest than CART. The idea behind choosing random forest is to have a overall better model accuracy although the model overfitting criteria has to be checked by testing the model on the hold out sample.  


# Create a subset of trainingValid data set
trainIndex <- createDataPartition(y = trainingValid$classe, p=0.5,list=FALSE) # 50% of the training sample is collected 
trainData <- trainingValid[trainIndex,]


Moreover, after creating this subset, I also removed the columns related to timestamps, the X columns, user_name, and new_window because they were not sensor values.They would not be much useful for prediction. This is done by executing the following code.

# Remove useless predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]

#> dim(trainData)
#[1] 9812   54

As a result, I had a subset of HAR data set that had only 9,812 rows of of 54 variables.

Then, based on the suggestion of the instructor (“… how you used cross validation”), I've used k-fold cross validation with K=4. As a rule of thum as I have 56 predictors hence for each sample if I should have at least 10 times of the number of the predictors as total number of observations in each sample i.e equaivalent to 540 observations. I could have choosen higher value of K to reduce the sample bias.   
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
                allowParallel = TRUE
                )
                
 I expected relatively good model performance, and a relatively low out of sample error rate. I would check that with the the help of testing sample.
 
 print(modFit)
 
 
#Random Forest 

#9812 samples
 # 53 predictor
  # 5 classes: 'A', 'B', 'C', 'D', 'E' 

#No pre-processing
#Resampling: Cross-Validated (4 fold) 

#Summary of sample sizes: 7358, 7361, 7359, 7358 

#Resampling results across tuning parameters:

 # mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
  # 2    0.9882803  0.9851720  0.0026245205  0.0033219497
  #27    0.9928656  0.9909745  0.0007839521  0.0009924234
  #53    0.9873627  0.9840122  0.0023983296  0.0030339991

#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was mtry = 27. 
 
 
 print(modFit$finalModel)

#Call:
 #randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
  #             Type of random forest: classification
   #                  Number of trees: 500
#No. of variables tried at each split: 27

 #       OOB estimate of  error rate: 0.4%
#Confusion matrix:
 #    A    B    C    D    E class.error
#A 2790    0    0    0    0 0.000000000
#B    6 1889    4    0    0 0.005265929
#C    0    8 1703    0    0 0.004675628
#D    0    2   12 1594    0 0.008706468
#E    0    1    0    6 1797 0.003880266

I observe that the accuracy of the model is great. I find a very few misclassification by glancing over the confusion matrix. i have to assure that the model is free from overfitting. I have to test this fact in the hold out testing data set. Then I will load test data set and check the model perfromance. 


# Load test data
testingAll = read.csv("pml-testing.csv",na.strings=c("NA",""))

# Only take the columns of testingAll that are also in trainData
testing <- testingAll[ , which(names(testingAll) %in% names(trainData))]

str(testing)

#'data.frame':	20 obs. of  53 variables:
# $ num_window          : int  74 431 439 194 235 504 485 440 323 664 ...
# $ roll_belt           : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
# $ pitch_belt          : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
# $ yaw_belt            : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
# $ total_accel_belt    : int  20 4 5 17 3 4 4 4 4 18 ...
# $ gyros_belt_x        : num  -0.5 -0.06 0.05 0.11 0.03 0.1 -0.06 -0.18 0.1 0.14 ...
# $ gyros_belt_y        : num  -0.02 -0.02 0.02 0.11 0.02 0.05 0 -0.02 0 0.11 ...
# $ gyros_belt_z        : num  -0.46 -0.07 0.03 -0.16 0 -0.13 0 -0.03 -0.02 -0.16 ...
# $ accel_belt_x        : int  -38 -13 1 46 -8 -11 -14 -10 -15 -25 ...
# $ accel_belt_y        : int  69 11 -1 45 4 -16 2 -2 1 63 ...
# $ accel_belt_z        : int  -179 39 49 -156 27 38 35 42 32 -158 ...
# $ magnet_belt_x       : int  -13 43 29 169 33 31 50 39 -6 10 ...
# $ magnet_belt_y       : int  581 636 631 608 566 638 622 635 600 601 ...
# $ magnet_belt_z       : int  -382 -309 -312 -304 -418 -291 -315 -305 -302 -330 ...
# $ roll_arm            : num  40.7 0 0 -109 76.1 0 0 0 -137 -82.4 ...
# $ pitch_arm           : num  -27.8 0 0 55 2.76 0 0 0 11.2 -63.8 ...
# $ yaw_arm             : num  178 0 0 -142 102 0 0 0 -167 -75.3 ...
# $ total_accel_arm     : int  10 38 44 25 29 14 15 22 34 32 ...
# $ gyros_arm_x         : num  -1.65 -1.17 2.1 0.22 -1.96 0.02 2.36 -3.71 0.03 0.26 ...
# $ gyros_arm_y         : num  0.48 0.85 -1.36 -0.51 0.79 0.05 -1.01 1.85 -0.02 -0.5 ...
# $ gyros_arm_z         : num  -0.18 -0.43 1.13 0.92 -0.54 -0.07 0.89 -0.69 -0.02 0.79 ...
# $ accel_arm_x         : int  16 -290 -341 -238 -197 -26 99 -98 -287 -301 ...
# $ accel_arm_y         : int  38 215 245 -57 200 130 79 175 111 -42 ...
# $ accel_arm_z         : int  93 -90 -87 6 -30 -19 -67 -78 -122 -80 ...
# $ magnet_arm_x        : int  -326 -325 -264 -173 -170 396 702 535 -367 -420 ...
# $ magnet_arm_y        : int  385 447 474 257 275 176 15 215 335 294 ...
# $ magnet_arm_z        : int  481 434 413 633 617 516 217 385 520 493 ...
# $ roll_dumbbell       : num  -17.7 54.5 57.1 43.1 -101.4 ...
# $ pitch_dumbbell      : num  25 -53.7 -51.4 -30 -53.4 ...
# $ yaw_dumbbell        : num  126.2 -75.5 -75.2 -103.3 -14.2 ...
# $ total_accel_dumbbell: int  9 31 29 18 4 29 29 29 3 2 ...
# $ gyros_dumbbell_x    : num  0.64 0.34 0.39 0.1 0.29 -0.59 0.34 0.37 0.03 0.42 ...
# $ gyros_dumbbell_y    : num  0.06 0.05 0.14 -0.02 -0.47 0.8 0.16 0.14 -0.21 0.51 ...
# $ gyros_dumbbell_z    : num  -0.61 -0.71 -0.34 0.05 -0.46 1.1 -0.23 -0.39 -0.21 -0.03 ...
# $ accel_dumbbell_x    : int  21 -153 -141 -51 -18 -138 -145 -140 0 -7 ...
# $ accel_dumbbell_y    : int  -15 155 155 72 -30 166 150 159 25 -20 ...
# $ accel_dumbbell_z    : int  81 -205 -196 -148 -5 -186 -190 -191 9 7 ...
# $ magnet_dumbbell_x   : int  523 -502 -506 -576 -424 -543 -484 -515 -519 -531 ...
# $ magnet_dumbbell_y   : int  -528 388 349 238 252 262 354 350 348 321 ...
# $ magnet_dumbbell_z   : int  -56 -36 41 53 312 96 97 53 -32 -164 ...
# $ roll_forearm        : num  141 109 131 0 -176 150 155 -161 15.5 13.2 ...
# $ pitch_forearm       : num  49.3 -17.6 -32.6 0 -2.16 1.46 34.5 43.6 -63.5 19.4 ...
# $ yaw_forearm         : num  156 106 93 0 -47.9 89.7 152 -89.5 -139 -105 ...
# $ total_accel_forearm : int  33 39 34 43 24 43 32 47 36 24 ...
# $ gyros_forearm_x     : num  0.74 1.12 0.18 1.38 -0.75 -0.88 -0.53 0.63 0.03 0.02 ...
# $ gyros_forearm_y     : num  -3.34 -2.78 -0.79 0.69 3.1 4.26 1.8 -0.74 0.02 0.13 ...
# $ gyros_forearm_z     : num  -0.59 -0.18 0.28 1.8 0.8 1.35 0.75 0.49 -0.02 -0.07 ...
# $ accel_forearm_x     : int  -110 212 154 -92 131 230 -192 -151 195 -212 ...
# $ accel_forearm_y     : int  267 297 271 406 -93 322 170 -331 204 98 ...
# $ accel_forearm_z     : int  -149 -118 -129 -39 172 -144 -175 -282 -217 -7 ...
# $ magnet_forearm_x    : int  -714 -237 -51 -233 375 -300 -678 -109 0 -403 ...
# $ magnet_forearm_y    : int  419 791 698 783 -787 800 284 -619 652 723 ...
# $ magnet_forearm_z    : int  617 873 783 521 91 884 585 -32 469 512 ...


# Run the prediction
pred <- predict(modFit, newdata = testing)

#> pred
 #[1] B A B A A E D B A A B C B A E E A B B B
#Levels: A B C D E

# Utility function provided by the instructor
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)

The model performed predictions very accurately, it correctly predicted 20 cases out of 20.

This leads to a  question that we have to test the model performance on a bigger sample. The test sample is too small.  





