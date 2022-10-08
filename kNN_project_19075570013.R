install.packages("class")
library(class)
install.packages("caret")
library(caret)

#read dataset
dataset<-read.csv("adult.csv", header=T, stringsAsFactors=T)

# to check the first 6 observations in the data
head(dataset) 

#EDA of the dependent variable
library(ggplot2)
barplot(table(dataset$income),main = 'Income Classification',col='blue',ylab ='No. of people')

#summary of the data
summary(dataset)

# This will replace '?' with 'NA'
dataset1<-read.csv("adult.csv",na.strings = c("?","NA"))

# Let's check summary again
summary(dataset1)
View(dataset1)
colSums(is.na(dataset1))

#-----RULES-----
install.packages("editrules")
library(editrules)
rules<- editfile("knn_rules.txt")
print(rules)
ve<-violatedEdits(rules,dataset1)
ve
summary(ve)
par(mar=c(3,3,3,3))
plot(ve)

#dropping columns
drop <- c("educational.num","marital.status","relationship","race","capital.loss","capital.gain","fnlwgt")
dataset1 = dataset1[,!(names(dataset1) %in% drop)]
View(dataset1)

#removing NA values
install.packages("VIM")
library(VIM)

# as it is observed only the following columns have NAs in them, we specifically perform kNN on these 3 variables
nrow(dataset1)
#We have taken k as 221 because sqrt of nrow(dataset1) comes as 221 approx.
dataset2<-kNN(dataset1,variable = c("workclass","occupation","native.country"),k=221)
# to verify if NAs removed
colSums(is.na(dataset2)) 

#now we create another data set excluding the dummy variables
dataset3<-dataset2[,1:10]

head(dataset3) # to verify if dummy variables removed
dim(dataset3) # gives the number of variables and columns in our dataset

#Lets check income with respect to age
library(ggplot2)
ggplot(dataset3) + aes(x=as.numeric(age), group=income, fill=income) + 
  geom_histogram(binwidth=1, color='black')+
  labs(x="Age",y="Count",title = "Income w.r.t Age")
#Lets check the same for workclass
barplot(table(dataset3$workclass),main = 'Income Classification w.r.t workclass',col='blue',ylab ='No. of people')

#Dividing data in Training and Testing Datasets
index<-createDataPartition(dataset3$age,p=0.75,list = F)
# argument 'list=F' is added so that it takes only indexes of the observations and not make a list row wise
train_adult<-dataset3[index,]
test_adult<-dataset3[-index,]
dim(train_adult)
dim(test_adult)

# model implementation
adult_blr<-glm(as.factor(income)~.,data = train_adult,family = "binomial") 
# argument (family = "binomial") is necessary as we are creating a model with dichotomous result

# To check how well our model is built we need to calculate predicted probabilities

train_adult$pred_prob_income<-fitted(adult_blr)
# this column will have predicted probabilities of being 1
head(train_adult) 
# run the command to check if the new column is added

# receiver operating characteristic
install.packages("ROCR")
library(ROCR)
# compares predicted values with actual values in training dataset
pred<-prediction(train_adult$pred_prob_income,train_adult$income)

# stores the measures with respect to which we want to plot the ROC graph
perf<-performance(pred,"tpr","fpr")

# plots the ROC curve
plot(perf,colorize=T,print.cutoffs.at=seq(0.1,by=0.05))

# we assign the threshold where sensitivity and specificity have almost similar values after observing the ROC graph
train_adult$pred_income<-ifelse(train_adult$pred_prob_income<0.3,0,1) 
# this column will classify probabilities we calculated and classify them as 0 or 1 based on our threshold value (0.3) and store in this column
head(train_adult)

#Creating confusion matrix and assessing the results:
table1<-table(train_adult$income,train_adult$pred_income)
table1
dim(train_adult)

test_adult$pred_prob_income<-predict(adult_blr,test_adult,type = "response")
# an extra argument(type = "response") is required while using 'predict' function to generate response as probabilities

test_adult$pred_income<-ifelse(test_adult$pred_prob_income<0.3,0,1)
# we take the same threshold to classify which we considered while classifying probabilities of training data
head(test_adult)
dim(test_adult)
table2<-table(test_adult$income,test_adult$pred_income)
table2

#accuracy
accuracy_train<-100*(sum(diag(table1))/sum(table1));
accuracy_train 
accuracy_test<-100*(sum(diag(table2))/sum(table2));
accuracy_test

#To check how much of our predicted values lie inside the curve:
auc<-performance(pred,"auc")
auc@y.values
