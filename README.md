# Genetic_classification

### Problem Statement: 
In this project, we aim to correctly classify genetic data with over 650 features. The features do not have inferential names and thus this is a pure machine learning problem.

### The overall strategy included the following steps:

1) Pre-Processing of data
2) Feature Selection
3) Scaling Data and spliting to train and test
4) Model building and finding right hyperparameters
5) Model stacking
6) Model testing and evaluation

### Pre-processing: 
Here we performed some basic steps to visualize the data and understand the underlying variations. We did the following:

Identified and deleted columns with only one unique value
Analyzed the columns and deleted a column if it had 1 unique
Identified any missing data and null values; we found none
Ensured that all object datatypes were converted into numeric datatypes
Identified categorical variables in the train set and one hot encoded them. We noticed that the test set had two additional one hot encoded columns thus we had to delete these columns so that the train and test set have the same number of columns
We then utilized correlations to detect high correlations within the 575 features and drop the features with a correlation of more than 0.95. This further reduced the number of features to around 332.
Scaled data using the min max scaler
Feature Selection and Division into Train and Validation Set: We used the BorutaPy method to selected supported features 30 for other classifiers. We used the random forest classifier within this method to rate features as per their importance.

This set of data was then split into train and validation sets for model building and comparison.

### Model Building: 
We built models for random forests, knn, XGboost, Gradient Boosting, SVC, Adabooster, ExtraTree and Neural Networks. For each of the models we utilized grid search to find the optimal parameters. We also used the StackingClassifier to combine the best random forest, knn, XGboost and GB methods. While the stacking classifier had a good accuracy, either the gradient boosting method or knn method alone outperformed the results.

### Model Comparison: 
We used the F1 score to compare results for each of the models. Since this is a very imbalanced dataset, using validation accuracy would be a very inapproriate measure. We noticed that for different sets of features(we tried models with 15, 46 and 100 features) gradient boost,knn and neural networks had the best performance.

### Model Stacking and Voting: 
Since the StackingClassifier did not give the best result, we created a voting method to utilize the probabilities of neural networks, gradient boosting and knn. This helped maximize the AUC for the final model and helped us achieve a score of 0.72151 on kaggle.

### Baseline Learning: 
The baseline learning from this project was that feature selection is a very important step in model building. The correct selection of features helps contribute the most to the predictions and having irrelavant features or highly correlated features reduced the accuracy of the models.
