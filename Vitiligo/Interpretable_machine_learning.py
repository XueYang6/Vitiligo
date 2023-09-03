import pandas as pd
import numpy as np
import matplotlib
import warnings

matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Matplotlib设置中文
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def rfc_model(x_train, x_test, y_train, param_grid=None):
    """
    use RandomForestClassifier to pred data
    :param x_train: x train data
    :param x_test: x test data
    :param y_train: y train data
    :return: y pred, y proba, rfc model
    """
    rfc = RandomForestClassifier(n_jobs=2)

    if param_grid is None:
        rfc.fit(x_train, y_train)
        y_pre = rfc.predict(x_test)
        y_proba = rfc.predict_proba(x_test)[:, 1]
    else:
        rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='roc_auc', cv=4)
        rfc_cv.fit(x_train, y_train)
        y_pre = rfc_cv.predict(x_test)
        y_proba = rfc_cv.predict_proba(x_test)[:, 1]
        rfc = rfc_cv

    return y_pre, y_proba, rfc


def xgb_model(x_train, x_test, y_train, param_grid=None):
    """
    use Xgboost to pred data
    :param x_train:
    :param x_test:
    :param y_train:
    :param param_grid:
    :return:
    """
    XGB = xgb.XGBClassifier()
    if param_grid is None:
        XGB.fit(x_train, y_train)
        y_pre = XGB.predict(x_test)
        y_proba = XGB.predict_proba(x_test)[:, 1]
    else:
        xgb_cv = GridSearchCV(estimator=XGB, param_grid=param_grid, scoring='roc_auc', cv=4)
        xgb_cv.fit(x_train, y_train)
        y_pre = xgb_cv.predict(x_test)
        y_proba = xgb_cv.predict_proba(x_test)[:, 1]
        XGB = xgb_cv
    return y_pre, y_proba, XGB


def svm_model(x_train, x_test, y_train, param_grid=None):
    """Use SVM to pred data"""
    SVM = svm.SVC(probability=True)
    if param_grid is None:
        SVM.fit(x_train, y_train)
        y_pre = SVM.predict(x_test)
        y_proba = SVM.predict_proba(x_test)[:, 1]
    else:
        svm_cv = GridSearchCV(estimator=SVM, param_grid=param_grid, scoring='roc_auc', cv=4)
        svm_cv.fit(x_train, y_train)
        y_pre = svm_cv.predict(x_test)
        y_proba = svm_cv.predict_proba(x_test)[:, 1]
        SVM = svm_cv
    return y_pre, y_proba, SVM


def gbdt_model(x_train, x_test, y_train, param_grid=None):
    """Use GBDT to pred data"""
    GBDT = GradientBoostingClassifier()
    if param_grid is None:
        GBDT.fit(x_train, y_train)
        y_pred = GBDT.predict(x_test)
        y_proba = GBDT.predict_proba(x_test)[:, 1]
    else:
        gbdt_cv = GridSearchCV(estimator=GBDT, param_grid=param_grid, scoring='roc_auc', cv=4)
        gbdt_cv.fit(x_train, y_train)
        y_pre = gbdt_cv.predict(x_test)
        y_proba = gbdt_cv.predict_proba(x_test)[:, 1]
        GBDT = gbdt_cv
    return y_pred, y_proba, GBDT


def draw_confusion_matrix(y_test, y_pre, display_labels, title):
    """
    confusion matrix
    :param y_test: y test
    :param y_pre:  y pred
    :param display_labels: class names
    :param title: graph title
    :return: plt.show() to show the graph
    """
    cm = confusion_matrix(y_test, y_pre, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(dpi=2000)
    disp.plot(cmap='Oranges', ax=ax)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.title(title)


def draw_roc(y_test, y_proba, title):
    """
    draw ROC
    :param y_test: y test values
    :param y_proba: y proba values
    :param title: graph title
    :return: plt.show() to show the graph
    """
    auc = metrics.roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds_test = metrics.roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_plot = ax.plot(fpr, tpr, color='black', lw=1.5, label='ROC curve (area = %0.4f)' % auc)
    ax.plot([0, 1], [0, 1], color='#d8d8d8', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.spines['top'].set_color('none')  # 将顶部边框线颜色设置为透明
    ax.spines['right'].set_color('none')  # 将右侧边框线颜色设置为透明
    plt.legend(loc="lower right")


def draw_all_roc(y_tests, probabilities, colors, labels, title):
    """
    draw_all_ROC
    :param y_tests: all y test values
    :param probabilities: all y proba values
    :param colors: lines colors
    :param labels: class names
    :return: plt.show() to show the graph
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=2000)
    for i in range(len(probabilities)):
        auc = metrics.roc_auc_score(y_tests[i], probabilities[i])
        fpr, tpr, thresholds_test = metrics.roc_curve(y_tests[i], probabilities[i])
        roc_plot = ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=labels[i] + ': ROC curve (AUC = %0.4f)' % auc)
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.plot([0, 1], [0, 1], color='#d8d8d8', lw=1.5, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.spines['top'].set_color('none')  # Set the top line color to transparent
    ax.spines['right'].set_color('none')  # Set the right line color to transparent
    ax.set_title(title)

class InterpretableMachineLearning:
    def __init__(self, df, category):
        self.df = df
        self.x = df.drop(columns=category)
        self.y = df[category]
        self.features = self.x.columns.values

    def dimension_reduce(self, method='lasso', visual=True):
        from sklearn.linear_model import Lasso, ElasticNet
        """
        Purpose: 使用Lasso或弹性网络来对数据进行降维

        Parameters: 
            1. method: choose 'lasso' or 'elastic net'
        """

        if method == 'lasso':
            lasso = Lasso(alpha=0.001, max_iter=1000)
            lasso.fit(self.x, self.y)
            coef = lasso.coef_
            lasso_features = []
            x_lasso = self.x.copy()
            for i in range(len(coef)):
                if coef[i] != 0:
                    lasso_features.append(self.features[i])  # choose features
                else:
                    x_lasso.drop(columns=self.features[i], inplace=True)  # delete columns where coef = 0
            # visual
            if visual:
                coef_values = pd.Series(coef, index=self.features).values
                plt.figure()
                plt.barh(self.features, coef_values)
                plt.title('the importance features')
                plt.show()
            return x_lasso, lasso_features

        elif method == 'elastic net':
            reg = ElasticNet(alpha=0.001, max_iter=1000)
            reg.fit(self.x, self.y)
            coef = reg.coef_
            reg_features = []
            x_reg = self.x.copy()
            for i in range(len(coef)):
                if coef[i] != 0:
                    reg_features.append(self.features[i])
                else:
                    x_reg.drop(columns=self.features[i], inplace=True)
            # visual
            if visual:
                coef_values = pd.Series(coef, index=self.features).values
                plt.figure()
                plt.barh(self.features, coef_values)
                plt.title('the importance features')
                plt.show()
            return x_reg, reg_features

    def model_prediction(self, x, y, model='RFC', param_grid=None, ROS=False):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        if ROS:
            ros = RandomOverSampler()
            x_train, y_train = ros.fit_resample(x_train, y_train)

        if model == 'RFC' or model == 'rfc':
            y_pre, y_proba, final_model = rfc_model(x_test=x_test, x_train=x_train, y_train=y_train,
                                                    param_grid=param_grid)
            auc = metrics.roc_auc_score(y_test, y_proba, multi_class='ovr')
            acc = metrics.accuracy_score(y_test, y_pre, normalize=True)

        elif model == 'XGB' or model == 'xgb':
            y_pre, y_proba, final_model = xgb_model(x_test=x_test, x_train=x_train, y_train=y_train,
                                                    param_grid=param_grid)
            auc = metrics.roc_auc_score(y_test, y_proba, multi_class='ovr')
            acc = metrics.accuracy_score(y_test, y_pre, normalize=True)

        elif model == 'SVM' or model == 'svm':
            y_pre, y_proba, final_model = svm_model(x_test=x_test, x_train=x_train, y_train=y_train,
                                                    param_grid=param_grid)
            auc = metrics.roc_auc_score(y_test, y_proba, multi_class='ovr')
            acc = metrics.accuracy_score(y_test, y_pre, normalize=True)

        elif model == 'GBDT' or model == 'gbdt':
            y_pre, y_proba, final_model = gbdt_model(x_test=x_test, x_train=x_train, y_train=y_train,
                                                     param_grid=param_grid)
            auc = metrics.roc_auc_score(y_test, y_proba, multi_class='ovr')
            acc = metrics.accuracy_score(y_test, y_pre, normalize=True)

        else:
            return warnings.warn("Please input right parameters")

        return y_train, y_test, y_pre, y_proba, acc, auc, final_model
