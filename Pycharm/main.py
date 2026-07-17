import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

dataset: pd.DataFrame = pd.read_csv("healthcare-dataset-stroke-data.csv")
dataset = dataset.drop('id', axis=1)
print(dataset.head())
print()
print(dataset.describe())
print('\n\n')

##################################################################
# Fill Missing Values
##################################################################

for columnName in dataset.select_dtypes(exclude='object').columns:
    column = dataset[columnName]
    column.fillna(round(column.mean()), inplace=True)

for columnName in dataset.select_dtypes(include='object').columns:
    column = dataset[columnName]
    column.fillna(column.mode()[0], inplace=True)

##################################################################
# Cast variables to correct type
##################################################################
dataset['age'] = dataset['age'].astype(int)
dataset['hypertension'] = dataset['hypertension'].astype(int)
dataset['heart_disease'] = dataset['heart_disease'].astype(int)
dataset['stroke'] = dataset['stroke'].astype(int)

##################################################################
# Encode Categorical Variables
##################################################################
labelEncoder = LabelEncoder()
for column in dataset.select_dtypes(include='object').columns:
    labelEncoder.fit(dataset[column])
    dataset[column] = labelEncoder.transform(dataset[column])

##################################################################
# Plot Variable Correlation Matrix
##################################################################
correlationMatrix = dataset.corr(numeric_only=True)
sns.heatmap(correlationMatrix, annot=True, cmap='Greens', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

##################################################################
# Plot Pair Plot of important variables
##################################################################
sns.pairplot(dataset[['age', 'avg_glucose_level', 'bmi']])
plt.show()

##################################################################
# Replace Outliers with Mean
##################################################################
for column in ['age', 'avg_glucose_level', 'bmi']:
    Q1 = dataset[column].quantile(0.25)
    Q2 = dataset[column].quantile(0.75)
    IQR = Q2 - Q1
    newMin = Q1 - 1.5 * IQR
    newMax = Q2 + 1.5 * IQR
    mean_value = dataset[column].mean()
    dataset[column] = dataset[column].where((dataset[column] >= newMin) & (dataset[column] <= newMax), mean_value)

##################################################################
# Normalize Data
##################################################################
minMaxNormalizer = MinMaxScaler(feature_range=(0, 1))
NumericalColumns = dataset[['age', 'avg_glucose_level', 'bmi']]
dataset[NumericalColumns.columns] = minMaxNormalizer.fit_transform(NumericalColumns)

##################################################################
# Getting Model Input and Output Data
##################################################################
datasetColumns = dataset.columns.tolist()
X = dataset[datasetColumns[:-1]]
Y = dataset[datasetColumns[-1]]


##################################################################
# Normal Train Test Splitting
##################################################################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

##################################################################
# KFold Train Test Splitting
##################################################################

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

##################################################################
# RANDOM FOREST
##################################################################

# Train Random Forest Classifier
randomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=42)
randomForestClassifier.fit(X_train, Y_train)

# Predict on the test set
yPredictedRF = randomForestClassifier.predict(X_test)

# Calculate accuracy
randomForestAccuracy = accuracy_score(Y_test, yPredictedRF)
print("Random Forest Accuracy:", randomForestAccuracy)

# Perform cross-validation
RandomForestCrossValidationScores = cross_val_score(randomForestClassifier, X, Y, cv=k_fold, scoring='accuracy')
print("Random Forest Mean Accuracy:", RandomForestCrossValidationScores.mean())
print("Random Forest Standard Deviation:", RandomForestCrossValidationScores.std())
print()


##################################################################
# XGBoost
##################################################################

# Train XGBoost classifier
xgbClassifier = xgb.XGBClassifier()
xgbClassifier.fit(X_train, Y_train)

# Predict on the test set
yPredictedXGB = xgbClassifier.predict(X_test)

# Calculate accuracy
xgbAccuracy = accuracy_score(Y_test, yPredictedXGB)
print("XGBoost Accuracy:", xgbAccuracy)

# Perform cross-validation
xgbCrossValidationScores = cross_val_score(xgbClassifier, X, Y, cv=k_fold, scoring='accuracy')
print("XGBoost Mean Accuracy:", xgbCrossValidationScores.mean())
print("XGBoost Standard Deviation:", xgbCrossValidationScores.std())
print()


##################################################################
# Logistic Regression
##################################################################

# Train Logistic Regression classifier
logRegClassifier = make_pipeline(StandardScaler(), LogisticRegression())
logRegClassifier.fit(X_train, Y_train)

# Predict on the test set
yPredictedLogReg = logRegClassifier.predict(X_test)

# Calculate accuracy
logRegAccuracy = accuracy_score(Y_test, yPredictedLogReg)
print("Logistic Regression Accuracy:", logRegAccuracy)

# Perform cross-validation
logRegCrossValidationScores = cross_val_score(logRegClassifier, X, Y, cv=k_fold, scoring='accuracy')
print("Logistic Regression Mean Accuracy:", logRegCrossValidationScores.mean())
print("Logistic Regression Standard Deviation:", logRegCrossValidationScores.std())
print()


##################################################################
# Comparing Models
##################################################################
ComparisonFrame = pd.DataFrame(data={'Model': ['Random Forest', 'XGBoost', 'Logistic Regression'],
                                     'Accuracy': [RandomForestCrossValidationScores.mean(),
                                                  xgbCrossValidationScores.mean(),
                                                  logRegCrossValidationScores.mean()],
                                     'Standard Deviation': [RandomForestCrossValidationScores.std(),
                                                            xgbCrossValidationScores.std(),
                                                            logRegCrossValidationScores.std()]})
sns.barplot(ComparisonFrame, x='Model', y='Accuracy')
plt.title('Accuracy Comparison of Models')
plt.show()

sns.barplot(ComparisonFrame, x='Model', y='Standard Deviation')
plt.title('Standard Deviation Comparison of Models')
plt.show()

randomForestFeatureImportance = randomForestClassifier.feature_importances_
sortedIndexes = np.argsort(randomForestFeatureImportance)
plt.barh(range(X.shape[1]), randomForestFeatureImportance[sortedIndexes], align='center')
plt.yticks(range(X.shape[1]), np.array(dataset.columns.tolist())[sortedIndexes])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()

xgb.plot_importance(xgbClassifier)
plt.title('XGBoost Feature Importance')
plt.show()

logRegModel = logRegClassifier.named_steps['logisticregression']
logRegCoefficients = np.abs(logRegModel.coef_[0])
sortedIndexes = np.argsort(logRegCoefficients)
plt.barh(np.array(dataset.columns.tolist()[:-1])[sortedIndexes], logRegCoefficients[sortedIndexes])
plt.xlabel('Absolute Coefficient Magnitude')
plt.ylabel('Feature')
plt.title('Logistic Regression Feature Importance')
plt.show()
