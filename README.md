# Practical_Machine_Learning- Project write up
In this project write-up, I have used the data from Human Activity Recognition (HAR). The aim was to train a model based on the data of 
various sensor values, which could later be used to predict the Classe variable, that is the manner in which the participants of HAR 
did the exercise.

After having examined the data briefly using the Rattle GUI, I have realized that some columns have a lot of missing (NA) values. 
Instead of trying to model them, I have decided to remove them from the data set. So the first step, after having loaded the 
required caret library (I've skipped the demonstration of Rattle GUI, since, after all, it was an interactive session with GUI part), 
was to detect and eliminate columns with a lot of missing values:
This resulted in 60 columns (variables), instead of 160.

After having removed the columns with missing values, I have proceeded to create a subset of the training data set because I have seen
that the whole set contained 19622 rows (observations) from the HAR study. 
I thought this was a lot of data because I wanted to use the Random Forests algorithm from the caret package, 
and my experience with it (when I used it for quizzes) indicated that it was a relatively expensive algorithm, my relatively old laptop 
computer spent a lot of CPU time and got heated for data sets that were much smaller than this HAR data set. 
Therefore I have decided to take 20% of the whole HAR data set as a representative sample. 

Moreover, after creating this subset, I also removed the columns related to timestamps, the X column, user_name, and new_window because 
they were not sensor values, so I thought they would not help much (or at all) for prediction:

As a result, I had a subset of HAR data set that had only 3927 rows of of 54 variables.

Then, based on the suggestion of the instructor (“… how you used cross validation”), I've decided to use cross validation, 
and instead of the usual 10-fold cross validation, I've used 4-fold cross validation 
(again, due to the limited resources of my machine, and my impatience, too). After setting the trainControl, I have finally 
used the Random Forests (rf) algorithm in the following manner:




