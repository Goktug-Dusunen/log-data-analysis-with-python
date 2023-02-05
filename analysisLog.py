import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
#Gd

data = pd.read_csv("log_data.csv")


X = data.drop("target", axis=1)
y = data["target"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=np.object).columns


numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])


param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5, 10]
}


grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)


accuracy = grid_search.score(X_test, y_test)
print("Test doğruluğu: {:.2f}%".format(accuracy * 100))
#Gd
