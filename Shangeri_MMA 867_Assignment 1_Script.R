library("readxl")
library("lubridate")
library("proto")
library("gsubfn")
library("RSQLite")
library("sqldf")
library("tidyverse")
library("ggplot2")
library("mice")
library("caret")
library("dplyr")
library("tidyr")
library("MASS")
library("car")
library("Metrics")
library("glmnet")
library("xgboost")
library("gbm")
library("mboost")

train <- read.csv("./Assignment//Individual/bike-sharing-demand//train.csv")
test  <- read.csv("./Assignment//Individual/bike-sharing-demand//test.csv")
#Duplicate the test data to save the datetime for later 
to.predict <- test

#step 1: lets look at the variables we are dealing with
head(train)
head(test)

#columns "casual" and "registered" is not required in the model, so let's remove that from the train
train = train[,!(names(train) %in% c("casual"))]
train = train[,!(names(train) %in% c("registered"))]

#Step 2: check for missing values
md.pattern(train)
md.pattern(test)
#no missing data!

#Step 3: Converting integer to factor
# #training set
train$season <- as.factor(train$season)
train$holiday <- as.factor(train$holiday)
train$workingday <- as.factor(train$workingday)
train$weather <- as.factor(train$weather)

#test set
test$season <- as.factor(test$season)
test$holiday <- as.factor(test$holiday)
test$workingday <- as.factor(test$workingday)
test$weather <- as.factor(test$weather)
# 
#Step 4: let's work with the train set
#Deriving day, hour from datetime field
train$datetime <- ymd_hms(train$datetime)
train$hour <- hour(train$date)
train$day <- wday(train$date)
train$month <- month(train$date, label=T)

#Deriving day, hour from datetime field
test$datetime <- ymd_hms(test$datetime)
test$hour <- hour(test$date)
test$day <- wday(test$date)
test$month <- month(test$date, label=T)

str(train) 
names(train)

str(test) 
names(test)

train[,11:13]<-lapply(train[,11:13], factor) #converting derived variables into factors
test[,10:12]<-lapply(test[,10:12], factor) #converting derived variables into factors

#Step 4: Removing datetime field 
train$datetime <- NULL
colnames(train)
str(train)


#Removing the data field for test
test$datetime <- NULL

#Step 5: visualization so we can understand the data
season_summary_by_hour <- sqldf('select season, hour, avg(count) as count from train group by season, hour')

#There are more rental in morning(from 7 hour) and evening(17-18th hour)
#People rent bikes more in Fall Summer and Winter, and much less in Spring
plot <- ggplot(train, aes(x=hour, y=count, color=season))+
  geom_point(data = season_summary_by_hour, aes(group = season))+
  geom_line(data = season_summary_by_hour, aes(group = season))+
  ggtitle("Bikes Rentals by Season")+ theme_minimal()+
  scale_colour_hue('Season',breaks = levels(train$season), 
                   labels=c('Spring', 'Summer', 'Fall', 'Winter'))
plot

#Step 6: separating train dataset to 70% for model building and 30% for testing
set.seed(1)
sample <- sample.int(n = nrow(train), size = floor(.7*nrow(train)), replace = F)
train_data <- train[sample, ]
train_target  <- train[-sample, ]

#Step 7: Start modeling with all factors
#build a model on training data using all variables (the 70%)
fit<-lm(count~., train_data) 
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(fit, train_target)
#score: 108.6011
rmse(train_target$count,predicted.rentals.testing)
    
#let's log(count)
logfit<-lm(log(count)~., train_data)
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(logfit, train_target)
predicted.rentals.testing.nonlog <- exp(predicted.rentals.testing)
#score: 0.613963
rmsle(train_target$count,predicted.rentals.testing.nonlog)

#using Step AIC
logfitAIC <- stepAIC(logfit, direction = 'both')
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(logfitAIC, train_target)
predicted.rentals.testing.nonlog <- exp(predicted.rentals.testing)
#score: 0.6140443 - ok this model is worst so we use log(count)~. for now
rmsle(train_target$count,predicted.rentals.testing.nonlog)

#let's use train in caret package with the log(count)~.
ctrl <- trainControl(method = "repeatedcv",
                     number = 3,
                     repeats = 3,
                     verboseIter = TRUE,
                     allowParallel = TRUE)
#xgbLinear from XGBoost 
logfit2 <- train(log(count)~., train_data,
                  method="xgbLinear", trControl = ctrl)
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(logfit2, train_target)
predicted.rentals.testing.nonlog <- exp(predicted.rentals.testing)
#score: 0.422968 - ok this model IS GOODDD
rmsle(train_target$count,predicted.rentals.testing.nonlog)

#LET'S PREDICT
predicted.rentals.final<-predict(logfit2, test)
predicted.rentals.final.nonlog <- exp(predicted.rentals.final)
predicted.rentals = data.frame(datetime = to.predict$datetime, count = predicted.rentals.final.nonlog)
colnames(predicted.rentals) <- c("datetime", "count")
write.csv(predicted.rentals, "Shangeri_MMA 867_Assignment 1_Predicted Rentals Solution.csv", row.names=FALSE)
#score on Kaggle 0.52418 [position 1902]

#attempt 2
#let's see restart and see if there is outliers
fit<-lm(count~., train) 
plot(density(resid(fit)))
sample_size <- nrow(train)
cooksd <- cooks.distance(fit)
#plot cook's distance
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")
#add cutoff line
abline(h = 4/sample_size, col="red")
#add labels
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4/sample_size, names(cooksd),""), col="red") 

#yes there is, remove them
outliers.located <- c(9000, 9009, 9010, 9008, 9011, 8334,
              8331, 8332, 8335, 8307, 8333, 8312,5662)
#remove the outliers and try modelling again
train <- train[-outliers.located, ]

#separating train dataset to 70% for model building and 30% for testing
set.seed(1)
sample <- sample.int(n = nrow(train), size = floor(.7*nrow(train)), replace = F)
train_data <- train[sample, ]
train_target  <- train[-sample, ]

#xgbLinear from XGBoost #new best!
logfit2 <- train(log(count)~., train_data,
                 method="xgbLinear", trControl = ctrl)
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(logfit2, train_target)
predicted.rentals.testing.nonlog <- exp(predicted.rentals.testing)
#score: 0.4237904 - ok this model looks better
rmsle(train_target$count,predicted.rentals.testing.nonlog)
#LET'S PREDICT
predicted.rentals.final<-predict(logfit2, test)
predicted.rentals.final.nonlog <- exp(predicted.rentals.final)
predicted.rentals = data.frame(datetime = to.predict$datetime, count = predicted.rentals.final.nonlog)
colnames(predicted.rentals) <- c("datetime", "count")
write.csv(predicted.rentals, "Shangeri_MMA 867_Assignment 1_Predicted Rentals Solution.csv", row.names=FALSE)
#score on Kaggle 0.50459 omgggg it got better!

#using stepAIC to find a better model
fit<-lm(log(count)~., train_data)
logfitAIC <- stepAIC(fit, direction = 'both')

#ok let's train
logfit2 <- train(log(count) ~ season + holiday + workingday + weather + temp + 
                    humidity + windspeed + hour + day + month, train_data,
                 method="xgbLinear", trControl = ctrl)
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(logfit2, train_target)
predicted.rentals.testing.nonlog <- exp(predicted.rentals.testing)
#score: 0.4231934 - ok this model is a little better but not good enough
rmsle(train_target$count,predicted.rentals.testing.nonlog)

#lets's log the season and hour
fit<-lm(log(count) ~ log(as.numeric(season)) + holiday + workingday + weather + temp + 
          humidity + windspeed + log(as.numeric(hour)) + day + month, train_data)
plot(fit)
#log helps, we get a normal curve
plot(density(resid(fit)))

#ok let's train again
logfit2 <- train(log(count) ~ log(as.numeric(season)) + holiday + workingday + weather + temp + 
                   humidity + windspeed + log(as.numeric(hour)) + day + month, train_data,
                 method="xgbLinear", trControl = ctrl)
#test model on the training target (30%) to see how good it is
predicted.rentals.testing<-predict(logfit2, train_target)
predicted.rentals.testing.nonlog <- exp(predicted.rentals.testing)
#score: 0.3432964 - ok this model IS GOODDD
rmsle(train_target$count,predicted.rentals.testing.nonlog)
#LET'S PREDICT
predicted.rentals.final<-predict(logfit2, test)
predicted.rentals.final.nonlog <- exp(predicted.rentals.final)
predicted.rentals = data.frame(datetime = to.predict$datetime, count = predicted.rentals.final.nonlog)
colnames(predicted.rentals) <- c("datetime", "count")
write.csv(predicted.rentals, "Shangeri_MMA 867_Assignment 1_Predicted Rentals Solution.csv", row.names=FALSE)
#score on Kaggle 0.49247 omggggggggg [position: 1446]