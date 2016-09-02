install.packages("fields")
install.packages("caret")
install.packages("ggplot2")
install.packages("RANN")
install.packages("rpart")
install.packages("rattle")
install.packages("randomForest")
install.packages("gbm")
install.packages("MASS")
install.packages("kernlab")

library(caret)
library(ggplot2)
library(fields)
library(RANN)
library(rpart)
library(rattle)
library(randomForest)
library(gbm)
library(MASS)
library(kernlab)

set.seed(1000)

#---------------------------------
#READING IN TRAIN AND TEST DATA

trainFileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testFileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainFileUrl), header=TRUE, quote="\"", na.strings=c("NA","#DIV/0!",""), stringsAsFactors=TRUE, strip.white=TRUE)
testing <- read.csv(url(testFileUrl), header=TRUE, quote="\"", na.strings=c("NA","#DIV/0!",""), stringsAsFactors=TRUE, strip.white=TRUE)

#check if the coloums are the same in testing and training data
col_nums_testing <- length(colnames(testing))-1
col_nums_training <- length(colnames(training))-1
all.equal(colnames(testing)[col_nums_testing], colnames(training)[col_nums_training])

dim(training)
dim(testing)

#---------------------------------
#PREPROCESSING

#throw out first 6 coloumns as they only contain metadata that cannot be used for prediction
training <- training[,-c(1:6)]
summary(training)

#throw out coloums where the percentage of missing data exceeds 50%
training <- training[, -which(colMeans(is.na(training)) > 0.5)]
dim(training)

#coloumn number of the variable we want to predict
col_num_classe <- which(colnames(training)=="classe")
col_num_classe

#rule out predictors that only have a single unique value
nzv_obj <- nearZeroVar(training[, -col_num_classe], saveMetrics = TRUE)
training <- training[, !nzv_obj$nzv]
dim(training)

#col_num_classe has to be defined again, if near zero variance predictors were found and due to that the dataset changed
col_num_classe <- which(colnames(training)=="classe")
col_num_classe

#find correlation between predictors
correlation <- findCorrelation(cor(training[, -col_num_classe]), cutoff=0.8)
names(training)[correlation]

#after the preprocessing following predictors are left
dim(training)
names(training)

#as we saw in the line 'names(training)[correlation]' many predictors are highly correlated, therefore PCA will be used in the traincontrol during preprocessing
#10 fold cross-validation is performed to split the training data into train and test sets.
#this ensures that the same transformations/preprocessing steps were performed on the train and test sets.
tc <- trainControl(method = "cv", number = 10, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)

#---------------------------------
#BUILDING PREDICTION MODELS


#1. RANDOM FOREST

rf <- train(classe ~ ., data = training, method = "rf", trControl= tc)
#accuracy of random forest model
max(rf$results$Accuracy)
#Kappa of random forest model
max(rf$results$Kappa)

#2. GBM - BOOSTING WITH TREES

gbm <- train(classe ~ ., data = training, method = "gbm", trControl= tc)
#accuracy of gbm model
max(gbm$results$Accuracy)
#Kappa of gbm model
max(gbm$results$Kappa)

#3 LINEAR DISCRIMINANT ANALYSIS
lda <- train(classe ~ ., data = training, method = "lda", trControl= tc)
#accuracy of lda model
max(lda$results$Accuracy)
#Kappa of lda model
max(lda$results$Kappa)

#3 SUPPORT VECTOR MACHINES
svmLinear <- train(classe ~ ., data = training, method = "svmLinear", trControl= tc)
#accuracy of svm model
max(svmLinear$results$Accuracy)
#Kappa of svm model
max(svmLinear$results$Kappa)


#---------------------------------

#PREDICTING WITH THE MODELS

rfPred <- predict(rf, testing)
rfPred

gbmPred <- predict(gbm, testing)
gbmPred



