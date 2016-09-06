install.packages("fields")
install.packages("caret", dependencies=TRUE)
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
training <- read.csv(url(trainFileUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testFileUrl), na.strings=c("NA","#DIV/0!",""))

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

#---------------------------------
#CREATING A CROSS VALIDATION DATASET

#now I split the dataset into two parts, one will be used for training (70%), the other part (30%) will be used for cross validation
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
trainingData <- training[inTrain, ]
crossValidationData <- training[-inTrain, ]


#---------------------------------
#BUILDING PREDICTION MODELS

#as we saw in the line 'names(training)[correlation]' many predictors are highly correlated, therefore PCA will be used in the traincontrol during preprocessing
#5 fold cross-validation is performed to split the training data into train and test sets.
tc <- trainControl(method = "cv", number = 5, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)

#1. RANDOM FOREST

rf <- train(classe ~ ., data = trainingData, method = "rf", trControl= tc)
rf

#2. GBM - BOOSTING WITH TREES

gbm <- train(classe ~ ., data = trainingData, method = "gbm", trControl= tc)
gbm

#3 LINEAR DISCRIMINANT ANALYSIS
lda <- train(classe ~ ., data = trainingData, method = "lda", trControl= tc)
lda

#4 SUPPORT VECTOR MACHINES
svmLinear <- train(classe ~ ., data = trainingData, method = "svmLinear", trControl= tc)
svmLinear


#---------------------------------
#OUT OF SAMPLE ERROR
#As we saw by looking at the confusionMatrix of the above models, the random forest and boosted trees have the highest Accuracy and Kappa values
#In this step I estimate the performance of the random forest and boosted trees models on the cross validation dataset.
predictRf <- predict(rf, crossValidationData)
confusionMatrix(crossValidationData$classe, predictRf)

accuracyRf <- postResample(predictRf, crossValidationData$classe)
accuracyRf

outOfSampleErrorRf <- 1 - as.numeric(confusionMatrix(crossValidationData$classe, predictRf)$overall[1])
outOfSampleErrorRf


predictGbm <- predict(gbm, crossValidationData)
confusionMatrix(crossValidationData$classe, predictGbm)

accuracyGbm <- postResample(predictGbm, crossValidationData$classe)
accuracyGbm

outOfSampleErrorGbm <- 1 - as.numeric(confusionMatrix(crossValidationData$classe, predictGbm)$overall[1])
outOfSampleErrorGbm


#---------------------------------

#PREDICTING WITH THE MODELS

rfPred <- predict(rf, testing)
rfPred

gbmPred <- predict(gbm, testing)
gbmPred