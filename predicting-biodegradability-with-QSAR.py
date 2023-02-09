#!/usr/bin/env python
# coding: utf-8

# # predicting-biodegradability-with-QSAR

# ### EDA

# In[96]:


#importing libraries and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[97]:


df = pd.read_csv('biodegradable_a.csv')
df.head()


# In[98]:


## Checking number of readily biodegradable (RB) vs non-readily biodegradable (NRB)
df.replace(['RB', 'NRB'], [0, 1], inplace=True)
vals, counts= np.unique(df["Biodegradable"], return_counts=True)
print("0 = RB, 1 = NRB")
for v, c in zip(vals, counts):
    print("# of ",v, "--->", c)


# In[99]:


df.info()


# In[100]:


pd.DataFrame(df.isna().sum()).transpose()


# In[101]:


#check for correlation
df_biodegradable_vars = df.loc[:,df.columns != 'Biodegradable']
df_biodegradable_y = df['Biodegradable']

df_biodegradable_corr = pd.concat([df_biodegradable_y, df_biodegradable_vars], axis=1)
corr = df_biodegradable_corr.corr()
corr_to_biodegradable = corr['Biodegradable'].sort_values(ascending=False)


# In[102]:


print(corr_to_biodegradable.head(10))


# In[103]:


df_biodegradable_vars.describe()


# In[104]:


df_biodegradable_vars.hist(bins=50, figsize=(20,15))
plt.show()


# ### Preprocessing the data:

# In[105]:


#Dropping columns with more than significant amount of missing values 
df_biodegradable_vars.drop('SpMax_B', axis=1, inplace=True)
df_biodegradable_vars.drop('C', axis=1, inplace=True)
df_biodegradable_vars.drop('nCp', axis=1, inplace=True)


# In[106]:


print(df_biodegradable_vars.isna().any(axis=1).sum())


# In[107]:


## Impute missing data and then normalise
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

trans_pipeline = Pipeline([
    ('simp_imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

features_transformed = trans_pipeline.fit_transform(df_biodegradable_vars)   #returns a transformed np array ready to be split


# In[108]:


pd.DataFrame(features_transformed).head()
y = df_biodegradable_y.to_numpy()


# In[109]:


print(pd.DataFrame(features_transformed).isna().any(axis=1).sum())


# ### Determining Feature Importances

# In[110]:


import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


rfx=RandomForestClassifier()
rfx.fit(features_transformed, y)

# Calculate feature importance in random forest
importances = rfx.feature_importances_
labels = df_biodegradable_vars.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"]).head(10)
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.figure(figsize=(50, 40), dpi=300)
plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.show()


# In[111]:


imortances = feature_df


# ### Splitting Data 

# In[112]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix,matthews_corrcoef, precision_score, recall_score

# create a stratified shuffle split object with a test size of 0.25
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
y = df_biodegradable_y.to_numpy()

# split the dataset into training and test sets
for train_index, test_index in stratified_split.split(features_transformed, y):
    X_train, X_test = features_transformed[train_index], features_transformed[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[113]:


from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def ClassStatistics(truth, preds):
    print(classification_report(truth, preds))
    print("MCC:", matthews_corrcoef(truth, preds))


# ### Training and optimizing models:

# In[114]:


#XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

params = {
    "learning_rate" : np.linspace(0.05, 0.4, 5),
    "max_depth" : [i for i in range(4, 20, 5)],
    "min_child_weight" : np.linspace(1, 8, 5),
    "gamma" : np.linspace(0.00, 0.4, 5),
    "n_estimators" : [i for i in range(5, 25, 5)],
    "colsample_bytree" : np.linspace(0.1, 0.8, 5),
}


boost = XGBClassifier()

random_search = RandomizedSearchCV(boost, param_distributions = params, n_iter = 5, scoring = "f1", cv = 5)

random_search.fit(X_train, y_train)

tuned_boost = random_search.best_estimator_     ## tuned XGBoost classifier 


# In[115]:


#grid_search = GridSearchCV(boost, params, scoring = "roc_auc", cv = 5, verbose =3)
#grid_search.fit(X_train, y_train)
#grid_search.best_estimator_


# In[116]:


#These were the parameters saved from the gridsearch. I copied them in to this cell to save us from running it again
#as the search took well over 15 minutes. 

#Added regularization (reg_alpha) to prevent overfitting 
gridtuned_boost = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                              colsample_bylevel=1, colsample_bynode=1,
                              colsample_bytree=0.45000000000000007, early_stopping_rounds=None,
                              enable_categorical=False, eval_metric=None, feature_types=None,
                              gamma=0.0, gpu_id=-1, grow_policy='depthwise',
                              importance_type=None, interaction_constraints='',
                              learning_rate=0.3125, max_bin=256, max_cat_threshold=64,
                              max_cat_to_onehot=4, max_delta_step=0, max_depth=9, max_leaves=0,
                              min_child_weight=1.0, monotone_constraints='()',
                              n_estimators=20, n_jobs=0, num_parallel_tree=1, predictor='auto',
                              random_state=0,reg_alpha = 0.5)

gridtuned_boost.fit(X_train,y_train)
print()


# In[117]:


##Random Forest Classifier

params = {
    "max_depth" : [i for i in range(4, 20, 5)],
    "n_estimators" : [i for i in range(5, 25, 5)],
}


rfc = RandomForestClassifier()

random_search = RandomizedSearchCV(rfc, param_distributions = params, n_iter = 5, scoring = "f1", cv = 5)
random_search.fit(X_train, y_train)

tuned_rfc = random_search.best_estimator_


# In[118]:


mdl_list = [tuned_boost, gridtuned_boost, tuned_rfc]
name = ["RS_XGBoost", "GS_XGBoost", "Random_Forest"]
idx=0

for i in mdl_list:
    scores = cross_val_score(i, X_train, y_train)
    print(name[idx] + ": " + str(np.mean(scores)))
    idx += 1


# ### Selecting Model & Evaluating Performance

# In[123]:


#Raandom Forst was ultimately selected
preds = tuned_rfc.predict(X_test)


# In[124]:


#Present Final Stats
final_stats = classification_report(y_test, preds, output_dict=True)
final_stats = pd.DataFrame(final_stats)
final_stats.columns = ['RB', 'NRB', 'accuracy', 'macro avg', 'weighted avg']
final_stats = final_stats.T


# In[125]:

final_stats


# In[126]:


final_MCC = "MCC: {}".format(matthews_corrcoef(y_test, preds))


# In[127]:


final_MCC

