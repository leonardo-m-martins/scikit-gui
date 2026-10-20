import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as me
from sklearn.preprocessing import StandardScaler
from src.enums import Methods


def metricas(texto, y_teste, y_pred):
    cm = me.confusion_matrix(y_teste, y_pred)
    ac = me.accuracy_score(y_teste, y_pred)
    pr = me.precision_score(y_teste, y_pred, average="macro")
    re = me.recall_score(y_teste, y_pred, average="macro")
    f1 = me.f1_score(y_teste, y_pred, average="macro")
    print("\n---------------- ", texto)
    print("Matriz de Confusão:\n", cm)
    print(f"Acurácia...........: {ac*100:.2f}%")
    print(f"Precisão...........: {pr*100:.2f}%")
    print(f"Recall.............: {re*100:.2f}%")
    print(f"F1-measure.........: {f1*100:.2f}%")


def ler_arquivo(arquivo):
    df = pd.read_csv(arquivo, header=None)

    atributos = df.iloc[:, :-1].astype(float).values.tolist()
    classes = df.iloc[:, -1].tolist()

    return atributos, classes


def mlp_treino(x_train, y_train):
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(125,),
        activation="logistic",
        solver="adam",
        alpha=0.01,
        learning_rate="adaptive",
        max_iter=10000,
    )
    mlp_model.fit(x_train, y_train)

    return mlp_model


def rf_treino(x_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(x_train, y_train)
    return rf_model


def decision_tree_training(feat_train, class_train):
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(feat_train, class_train)
    return model


def ada_boost_training(feat_train, class_train):
    model = AdaBoostClassifier(random_state=42)
    model.fit(feat_train, class_train)
    return model


SUPPORTED_MODELS = (
    MLPClassifier | RandomForestClassifier | DecisionTreeClassifier | AdaBoostClassifier
)


def test_model(features, classes, model: SUPPORTED_MODELS, name: str):
    pred = model.predict(features)
    metricas(name, classes, pred)


def run_individual_method(filename: str, split, method: Methods):
    features, classes_str = ler_arquivo(filename)
    classes = list(set(classes_str))
    classes = np.array([classes.index(c) for c in classes_str])
    features_training, features_test, classes_training, classes_test = train_test_split(
        features, classes, test_size=split / 100, random_state=42, stratify=classes
    )

    match method:
        case Methods.RANDOM_FOREST:
            train_model = rf_treino
        case Methods.MULTILAYER_PERCEPTRON:
            scaler = StandardScaler()
            features_training = scaler.fit_transform(features_training)
            features_test = scaler.fit_transform(features_test)
            train_model = mlp_treino
        case Methods.PART:
            train_model = decision_tree_training
        case Methods.ADABOOSTM1:
            train_model = ada_boost_training
        case _:
            print(f"Método {method} não suportado!")
            return

    model = train_model(features_training, classes_training)
    test_model(features_test, classes_test, model, method)


def run_all_methods(filename: str, split: int):
    for method in Methods:
        run_individual_method(filename, split, method)
