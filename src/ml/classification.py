from sklearn import metrics as me
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods, Preprocessors
from src.ml.common import ModeConfig, ModelSpec
from src.shared import ClassificationMetrics

_MODEL_SPECS: dict[Methods, ModelSpec] = {
    Methods.MULTILAYER_PERCEPTRON: ModelSpec(
        MLPClassifier,
        preprocessing=Preprocessors.STANDARD_SCALER,
    ),
    Methods.RANDOM_FOREST: ModelSpec(RandomForestClassifier),
    Methods.DECISION_TREE: ModelSpec(DecisionTreeClassifier),
    Methods.ADABOOSTM1: ModelSpec(AdaBoostClassifier),
    Methods.GAUSSIAN_NAIVE_BAYES: ModelSpec(GaussianNB),
    Methods.MULTI_NAIVE_BAYES: ModelSpec(MultinomialNB),
    Methods.COMPLEMENT_NAIVE_BAYES: ModelSpec(ComplementNB),
    Methods.LOGISTICS: ModelSpec(
        LogisticRegression,
        preprocessing=Preprocessors.STANDARD_SCALER,
    ),
    Methods.SVC: ModelSpec(SVC, preprocessing=Preprocessors.STANDARD_SCALER),
    Methods.K_NEIGHBORS: ModelSpec(
        KNeighborsClassifier, preprocessing=Preprocessors.STANDARD_SCALER
    ),
    Methods.GRADIENT_BOOSTING: ModelSpec(GradientBoostingClassifier),
    Methods.HIST_GRADIENT_BOOST: ModelSpec(HistGradientBoostingClassifier),
}


def classification_metrics(y_true, y_pred, labels=None) -> ClassificationMetrics:
    if labels is not None:
        present = set(y_true) | set(y_pred)
        labels = [label for label in labels if label in present]

    cm = me.confusion_matrix(y_true, y_pred, labels=labels)
    ac = me.accuracy_score(y_true, y_pred)
    pr = me.precision_score(y_true, y_pred, average="macro", labels=labels)
    re = me.recall_score(y_true, y_pred, average="macro", labels=labels)
    f1 = me.f1_score(y_true, y_pred, average="macro", labels=labels)
    return ClassificationMetrics(ac, pr, re, f1, cm)


CLASSIFICATION_CONFIG = ModeConfig(_MODEL_SPECS, classification_metrics)
