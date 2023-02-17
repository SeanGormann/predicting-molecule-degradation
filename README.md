# predicting-molecule-degradation
Here I'm attempting to predict the biodegradability of molecules using Quantitive Structure Activity Relationship (QSAR) data. The data was obtained through UCI and is contained in the file biodegradable_a.csv. After loading the dataset in and performing some exploratory data anlalysis, the most important features are then determined using a random forest. I've also been experimenting with implementing with continuous integration (CI) in my projects and so every time a new push to the main branch is committed, a report with this feature importance graph (and later some statistics about the final model) is generated. 

Feature Importances:


![feature_importance](https://user-images.githubusercontent.com/100109163/219794682-9b3db22d-07f9-4a31-8f90-b10ded411d99.png)



I decided to leave the majority of features in when making predictions. I tested several models including XGBoost with two different methods of hyperparameter searching (GridSearchCV and RandomSearchCV). F1 score was used during cross validation to select the optimal model and some final statistics were generated giving the model's accuracy.

