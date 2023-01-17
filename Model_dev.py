import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import NuSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNetCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


class RegressionModels():

    def __init__(self, X_training, y_training, X_test, y_test):
        self.Xtrain = X_training
        self.ytrain = y_training
        self.Xtest = X_test
        self.ytest = y_test
        self.results_r2 = []
        self.results_MSE = []
        self.results_MAE = []
        self.results_SMSE = []

    def fit_models(self):
        self.M1 = LinearRegression()
        self.M2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        self.M3 = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        self.M4 = Ridge()
        self.M5 = make_pipeline(PolynomialFeatures(degree=2), Ridge())
        self.M6 = make_pipeline(PolynomialFeatures(degree=3), Ridge())
        self.M7 = Lasso()
        self.M8 = make_pipeline(PolynomialFeatures(degree=2), Lasso())
        self.M9 = make_pipeline(PolynomialFeatures(degree=3), Lasso())
        self.M10 = SGDRegressor(max_iter=2000, tol=1e-3)
        self.M11 = DecisionTreeRegressor()
        self.M12 = RandomForestRegressor()
        self.M13 = AdaBoostRegressor()
        self.M14 = GradientBoostingRegressor()
        self.M15 = KNeighborsRegressor(n_neighbors=2)
        self.M16 = SVR()
        self.M17 = MLPRegressor(hidden_layer_sizes=(200,))
        self.M18 = HistGradientBoostingRegressor()
        self.M19 = XGBRFRegressor()
        self.M20 = LGBMRegressor()
        # self.M21 = CatBoostRegressor(verbose=0)
        self.M22 = BaggingRegressor()
        self.M23 = NuSVR()
        self.M24 = ExtraTreesRegressor()
        self.M25 = PoissonRegressor()
        self.M26 = HuberRegressor()
        self.M27 = RidgeCV()
        self.M28 = BayesianRidge()
        self.M29 = make_pipeline(PolynomialFeatures(degree=2), ElasticNetCV())
        self.M30 = TransformedTargetRegressor(regressor=self.M29)
        self.M31 = TweedieRegressor()
        self.M32 = RANSACRegressor()
        self.M33 = OrthogonalMatchingPursuitCV()
        self.M34 = PassiveAggressiveRegressor()
        self.Name = ['Linear Regression', 'Linear Regression digree 2', 'Linear Regression digree 3',
                     'Ridge Regression',
                     'Ridge Regression digree 2', 'Ridge Regression digree 3', 'Lasso', 'Lasso digree 2',
                     'Lasso digree 3',
                     'SGDRegressor',
                     'Decision Tree Regressor', 'Random Forest Regressor', 'AdaBoost Regressor',
                     'Gradient Boosting Regressor', 'KNN Regressor',
                     'Support Vector Machine', 'Neural Network Regression', 'Histogram Boosting', 'XGB Boosting',
                     'Light GBM',
                     'BaggingRegressor', 'NuSVR', 'ExtraTreesRegressor', 'PoissonRegressor', 'HuberRegressor',
                     'RidgeCV', 'BayesianRidge', 'ElasticNetCV', 'TransformedTargetRegressor', 'TweedieRegressor',
                     'RANSACRegressor', 'OrthogonalMatchingPursuitCV', 'PassiveAggressiveRegressor']
        self.clfs = [self.M1, self.M2, self.M3, self.M4, self.M5, self.M6, self.M7, self.M8, self.M9, self.M10,
                     self.M11, self.M12, self.M13, self.M14, self.M15, self.M16, self.M17, self.M18, self.M19,
                     self.M20, self.M22, self.M23, self.M24, self.M25, self.M26, self.M27, self.M28,
                     self.M29, self.M30, self.M31, self.M32, self.M33, self.M34]

        for i in self.clfs:
            i.fit(self.Xtrain, self.ytrain)
            self.r2 = r2_score(self.ytest, i.predict(self.Xtest))
            self.MSE = mean_squared_error(self.ytest, i.predict(self.Xtest))
            self.MAE = mean_absolute_error(self.ytest, i.predict(self.Xtest))
            self.SMSE = np.sqrt(mean_absolute_error(self.ytest, i.predict(self.Xtest)))
            self.results_r2.append(self.r2)
            self.results_MSE.append(self.MSE)
            self.results_MAE.append(self.MAE)
            self.results_SMSE.append(self.SMSE)

        self.dict = {'R2': self.results_r2, 'MSE': self.results_MSE, 'MAE': self.results_MAE, 'SMSE': self.results_SMSE}
        self.score = pd.DataFrame(self.dict, index=self.Name)

        return self.score



