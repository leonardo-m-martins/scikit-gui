import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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


def mlp_teste(x_teste, y_teste, mlp_model):
    y_pred = mlp_model.predict(x_teste)
    metricas("MLP", y_teste, y_pred)


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


def rf_teste(x_teste, y_teste, rf_model):

    y_pred = rf_model.predict(x_teste)
    metricas("RANDOM FOREST", y_teste, y_pred)

def test_model(features, classes, model: MLPClassifier | RandomForestClassifier, name: str):
    pred = model.predict(features)
    metricas(name, classes, pred)

def run_individual_method(filename: str, split, method: Methods):
    features, classes_str = ler_arquivo(filename)
    classes = list(set(classes_str))
    classes = np.array([classes.index(c) for c in classes_str])
    features_training, features_test, classes_training, classes_test = train_test_split(
        features, classes, test_size=split/100, random_state=42, stratify=classes
    )

    match method:
        case Methods.RANDOM_FOREST:
            train_model = rf_treino
        case Methods.MULTILAYER_PERCEPTRON:
            scaler = StandardScaler()
            features_training = scaler.fit_transform(features_training)
            features_test = scaler.fit_transform(features_test)
            train_model = mlp_treino
        case _:
            print(f"Método {method} não suportado!")
            return

    model = train_model(features_training, classes_training)
    test_model(features_test, classes_test, model, method)

def run_all_methods(filename: str, split: int):
    for method in Methods:
        run_individual_method(filename, split, method)

# **************************************************************
#                       MÓDULO PRINCIPAL
# **************************************************************
if __name__ == "__main__":
    arquivo = "data/spambase.data"
    feat, cl_str = ler_arquivo(arquivo)
    classes = list(set(cl_str))
    cl = np.array([classes.index(c) for c in cl_str])
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat, cl, test_size=0.2, random_state=42, stratify=cl
    )

    # NORMALIZAÇÃO PARA MLP
    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)

    # MLP
    modelo_mlp = mlp_treino(x_treino_sc, y_treino)
    mlp_teste(x_teste_sc, y_teste, modelo_mlp)

    # RANDOM FOREST
    modelo_rf = rf_treino(x_treino, y_treino)
    rf_teste(x_teste, y_teste, modelo_rf)
