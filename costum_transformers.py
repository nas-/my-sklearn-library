from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class DegreesValuesFixer(BaseEstimator, TransformerMixin):
    # trasforms values in degrees in their sinus.
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


from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin


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
