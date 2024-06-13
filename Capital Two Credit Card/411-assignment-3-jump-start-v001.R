.rs.restartR()

install.packages('farff')
#install.packages('robustbase')
install.packages('cvTools')
install.packages('car')
install.packages('robustbase')
install.packages('caret')
install.packages("rlang")

setwd('/Users/ureemjames/Desktop/MSDS 411/Assignments/Module 3')

library(farff) # for reading arff file
library(robustbase)
library(cvTools) # explicit creation of folds for cross-validation
library(ModelMetrics) # used for precision-recall evaluation of classifiers
library(car) # for recode function
library(caret)



# optimal cutoff for predicting bad credit set as
# (cost of false negative/cost of false positive) times
# (prevalence of positive/prevalence of negative)
# (1/5)*(.3/.7) = 0.086
CUTOFF = 0.086
COSTMATRIX = matrix(c(0,5,1,0), nrow = 2, ncol = 2, byrow = TRUE)

COSTMATRIX

credit = readARFF("dataset_31_credit-g.arff")
# write to comma-delimited text for review in Excel
write.csv(credit, file = "credit.csv", row.names = FALSE)

# check structure of the data frame
cat("\n\nStucture of initial credit data frame:\n")
print(str(credit))

# quick summary of credit data
cat("\n\nSummary of initial credit data frame:\n")
print(summary(credit))

# personal_status has level "female single" with no observations
cat("\n\nProblems with personal_status, no single females:\n")
print(table(credit$personal_status))
# fix this prior to analysis
credit$personal_status = factor(as.numeric(credit$personal_status),
    levels = c(1,2,3,4), 
    labels = c("male div/sep","female div/dep/mar","male single","male mar/wid"))
print(table(credit$personal_status))

cat("\n\nProblems with purpose, low- and no-frequency levels:\n")
print(table(credit$purpose))
# keep first four classes: "new car", "used car", "furniture/equipment", "radio/tv"
# keep "education" and "business" with new values 
# add "retraining" to "education"
# gather all other levels into "other"
credit$purpose = recode(credit$purpose, '"new car" = "new car";
    "used car" = "used car"; 
    "furniture/equipment" = "furniture/equipment";
    "radio/tv" = "radio/tv"; 
    "education" = "education"; "retraining" = "education";
    "business" = "business"; 
    "domestic appliance" = "other"; "repairs" = "other"; "vacation" = "other"; 
    "other" = "other" ',
    levels = c("new car","used car","furniture/equipment","radio/tv", 
    "education","business","other" ))

cat("\n\nPurpose variable re-leveled:\n")
print(table(credit$purpose))

# credit_amount is highly skewed... use log_credit_amount instead
credit$log_credit_amount = log(credit$credit_amount)    

# summary of transformed credit data
cat("\n\nSummary of revised credit data frame:\n")
print(summary(credit))

str(credit)

print(table(credit$class))

print(table(credit$foreign_worker))

credit$class

# write.csv(credit, file = "postCapitalTwo.csv", row.names = FALSE)

head(credit,3)

# logistic regression evaluated with cross-validation
# include explanatory variables except foreign_worker
# (only 37 of 100 cases are foreign workers)
credit_model = "class ~ checking_status + duration + 
    credit_history + purpose + log_credit_amount + savings_status + 
    employment + installment_commitment + personal_status +        
    other_parties + residence_since + property_magnitude +
    age + other_payment_plans + housing + existing_credits +      
    job + num_dependents + own_telephone" 

set.seed(1)
nfolds = 5
folds = cvFolds(nrow(credit), K = nfolds) # creates list of indices

baseprecision = rep(0, nfolds)  # precision with 0 cutoff
baserecall = rep(0, nfolds)  # recall with  0 cutoff
basef1Score = rep(0, nfolds)  # f1Score with 0 cutoff
basecost = rep(0, nfolds)  # total cost with 0 cutoff
ruleprecision = rep(0, nfolds)  # precision with CUTOFF rule
rulerecall = rep(0, nfolds)  # recall with CUTOFF rule
rulef1Score = rep(0, nfolds)  # f1Score with CUTOFF rule
rulecost = rep(0, nfolds)  # total cost with CUTOFF rule

#folding starts here: 
for (ifold in seq(nfolds)) 
  {
    # cat("\n\nSUMMARY FOR IFOLD:", ifold) # checking in development
    # print(summary(credit[(folds$which == ifold),]))
    # train model on all folds except ifold
    train = credit[(folds$which != ifold), ]
    test = credit[(folds$which == ifold),]
    credit_fit = glm(credit_model, family = binomial,
        data = train)
    # evaluate on fold ifold    
    credit_predict = predict.glm(credit_fit, 
        newdata = test, type = "response") 
    baseprecision[ifold] = ppv(as.numeric(test$class)-1, 
        credit_predict, cutoff = 0.5)  
    baserecall[ifold] = ModelMetrics::recall(as.numeric(test$class)-1, 
        credit_predict, cutoff = 0.5) 
    basef1Score[ifold] = f1Score(as.numeric(test$class)-1, 
        credit_predict, cutoff = 0.5) 
    basecost[ifold] = sum(
      ModelMetrics::confusionMatrix(as.numeric(test$class)-1,
        credit_predict) * COSTMATRIX)  
    ruleprecision[ifold] = ppv(as.numeric(test$class)-1, 
        credit_predict, cutoff = CUTOFF)  
    rulerecall[ifold] = ModelMetrics::recall(as.numeric(test$class)-1, 
        credit_predict, cutoff = CUTOFF) 
    rulef1Score[ifold] = f1Score(as.numeric(test$class)-1, 
        credit_predict, cutoff = CUTOFF)
    rulecost[ifold] = sum(
      ModelMetrics::confusionMatrix(as.numeric(test$class)-1, 
            credit_predict,cutoff=CUTOFF) * COSTMATRIX)                                    
  } 
cvbaseline = data.frame(baseprecision, baserecall, basef1Score, basecost,
    ruleprecision, rulerecall, rulef1Score, rulecost)

#Testing Code
#ModelMetrics::recall(as.numeric(test$class)-1, credit_predict, cutoff = 0.5)
#ppv(as.numeric(test$class)-1, credit_predict, cutoff = 0.5)
#f1Score(as.numeric(test$class)-1, credit_predict, cutoff = 0.5)
#ModelMetrics::confusionMatrix(as.numeric(test$class)-1,credit_predict) * COSTMATRIX




cat("\n\nCross-validation summary across folds:\n")
print(round(cvbaseline, digits = 3))



# This is where we were last time. 

cat("\n\nCross-validation baseline results under cost cutoff rules:")
cat("\n    Precision: ", round(mean(cvbaseline$ruleprecision), digits = 3))
cat("\n    Recall: ", round(mean(cvbaseline$rulerecall), digits = 3))
cat("\n    F1 Score: ", round(mean(cvbaseline$rulef1Score), digits = 3))
cat("\n    Average cost per fold: ", round(mean(cvbaseline$rulecost), digits = 2), "\n")


# Added Code Starts Here: 

encoded_data = read.csv('encoded.csv', stringsAsFactors = TRUE)
head(encoded_data, 3)

str(encoded_data)

new_model = "class ~ labels"

set.seed(1)
nfolds = 5
folds = cvFolds(nrow(encoded_data), K = nfolds) # creates list of indices

baseprecision = rep(0, nfolds)  # precision with 0 cutoff
baserecall = rep(0, nfolds)  # recall with  0 cutoff
basef1Score = rep(0, nfolds)  # f1Score with 0 cutoff
basecost = rep(0, nfolds)  # total cost with 0 cutoff
ruleprecision = rep(0, nfolds)  # precision with CUTOFF rule
rulerecall = rep(0, nfolds)  # recall with CUTOFF rule
rulef1Score = rep(0, nfolds)  # f1Score with CUTOFF rule
rulecost = rep(0, nfolds)



for (ifold in seq(nfolds)) 
{
  # cat("\n\nSUMMARY FOR IFOLD:", ifold) # checking in development
  # print(summary(credit[(folds$which == ifold),]))
  # train model on all folds except ifold
  train = encoded_data[(folds$which != ifold), ]
  test = encoded_data[(folds$which == ifold),]
  credit_fit = glm(new_model, family = binomial,
                   data = train)
  # evaluate on fold ifold    
  credit_predict = predict.glm(credit_fit, 
                               newdata = test, type = "response") 
  baseprecision[ifold] = ppv(as.numeric(test$class)-1, 
                             credit_predict, cutoff = 0.5)  
  baserecall[ifold] = ModelMetrics::recall(as.numeric(test$class)-1, 
                                           credit_predict, cutoff = 0.5) 
  basef1Score[ifold] = f1Score(as.numeric(test$class)-1, 
                               credit_predict, cutoff = 0.5) 
  basecost[ifold] = sum(
    ModelMetrics::confusionMatrix(as.numeric(test$class)-1,
                                  credit_predict) * COSTMATRIX)  
  ruleprecision[ifold] = ppv(as.numeric(test$class)-1, 
                             credit_predict, cutoff = CUTOFF)  
  rulerecall[ifold] = ModelMetrics::recall(as.numeric(test$class)-1, 
                                           credit_predict, cutoff = CUTOFF) 
  rulef1Score[ifold] = f1Score(as.numeric(test$class)-1, 
                               credit_predict, cutoff = CUTOFF)
  rulecost[ifold] = sum(
    ModelMetrics::confusionMatrix(as.numeric(test$class)-1, 
                                  credit_predict,cutoff=CUTOFF) * COSTMATRIX)                                    
} 
cvaltapproach = data.frame(baseprecision, baserecall, basef1Score, basecost,
                        ruleprecision, rulerecall, rulef1Score, rulecost)

cat("\n\nCross-validation summary across folds:\n")
print(round(cvaltapproach, digits = 3))




cat("\n\nCross-validation baseline results under cost cutoff rules:")
cat("\n    Precision: ", round(mean(cvaltapproach$ruleprecision), digits = 3))
cat("\n    Recall: ", round(mean(cvaltapproach$rulerecall), digits = 3))
cat("\n    F1 Score: ", round(mean(cvaltapproach$rulef1Score), digits = 3))
cat("\n    Average cost per fold: ", round(mean(cvaltapproach$rulecost), digits = 2), "\n")





#Extra Code Below that is not used.
# prepare data for input to autoencoder work
#design_matrix = model.matrix(as.formula(credit_model), data = credit)
#design_data_frame = as.data.frame(design_matrix)[,-1]  # dropping the intercept term
# normalize the data 
#minmaxnorm <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
#minmax_data_frame <- lapply(design_data_frame, FUN = minmaxnorm)

#cat("\n\nStructure of minmax_data_frame for input to autoencoding work:\n")
#print(str(minmax_data_frame))