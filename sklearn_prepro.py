from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector


def produce_model():
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown="ignore"))])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude=["category", 'object', 'bool'])),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=["float", 'int', 'bool'])),
        ], remainder='passthrough')
    return Pipeline(
        steps=[("preprocessor", preprocessor)]
    )


model = clf['model']
kfold = KFold(n_splits=num_folds, random_state=seed)
cv_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring, verbose=0)

model.fit(X_train, Y_train)
prediction = model.predict(X_test)
score = model.score(X_test, Y_test)
clf['test_score'] = score
print(clf['description'].rjust(30), f'train = {cv_scores.mean():.3f} ({cv_scores.std():.3f}), test = {score:.3f}\n')