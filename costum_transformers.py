import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class DegreesValuesFixer(BaseEstimator, TransformerMixin):
    # Transform values in degrees in their sinus.
    # Useful because 365 is far from 0, but sin365 is close to sin0
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # we change all values to radiants and convert to the sinus
        # X[X<0]+=360
        # X[X>=360]-=360
        X = np.sin(np.radians(X))
        return X


class SubValueTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, min_val, max_val):
        # trasforms values that are higher than max to max and less then min to min
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[X < self.min_val] = self.min_val
        X[X > self.max_val] = self.max_val
        return X


class PseudoLabeler(BaseEstimator, RegressorMixin):
    """
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    # model_factory = [
    #  XGBRegressor(nthread=1),
    #  PseudoLabeler(
    #  XGBRegressor(nthread=1),
    #  test,
    #  features,
    #  target,
    #  sample_rate=0.3
    #  ),
    # ]
    #
    # for model in model_factory:
    #  model.seed = 42
    #  num_folds = 8
    #
    #  scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error', n_jobs=8)
    #  score_description = "MSE: %0.4f (+/- %0.4f)" % (np.sqrt(scores.mean()*-1), scores.std() * 2)
    #
    # print('{model:25} CV-{num_folds} {score_cv}'.format(
    #  model=model.__class__.__name__,
    #  num_folds=num_folds,
    #  score_cv=score_description
    #  ))
    """

    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        """
        @sample_rate - percent of samples used as pseudo-labelled data
        from the unlabelled dataset
        @url=https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/
        """
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Fit the data using pseudo labeling.
        """
        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[self.features],
            augemented_train[self.target]
        )
        return self

    def __create_augmented_train(self, X, y):
        """
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        """
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])
        return shuffle(augemented_train)

    def predict(self, X):
        """
        Returns the predicted values.
        """
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__


class BoostedHybrid(BaseEstimator, RegressorMixin):
    """
    #@url=https://www.kaggle.com/teckmengwong/tps2201-hybrid-time-series/notebook#Hybrid-Models
    Linear regression excels at extrapolating trends, but can't learn interactions.
    XGBoost excels at learning interactions, but can't extrapolate trends.
    Apply the Second to the residual of the first, so to combine them
    """

    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        # Train model_1
        self.model_1.fit(X, y)

        # Make predictions
        y_fit = self.model_1.predict(X)
        # Compute residuals
        y_resid = y - y_fit

        # Train model_2 on residuals , eval_set=[(X_1_valid, y_valid_resid)]
        self.model_2.fit(X, y_resid)
        # Model2 prediction
        y_fit2 = self.model_2.predict(X)
        # Compute noise
        y_resid2 = y_resid - y_fit2

        # Save data for question checking
        self.y = y
        self.y_fit = y_fit
        self.y_resid = y_resid
        self.y_fit2 = y_fit2
        self.y_resid2 = y_resid2

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        # Predict with model_1
        y_predict = self.model_1.predict(X)
        # Add model_2 predictions to model_1 predictions
        y_predict += self.model_2.predict(X)

        return y_predict

    def get_model_name(self):
        self.__name__ = f'{self.model_1.__class__.__name__}_{self.model_2.__class__.__name__}'
