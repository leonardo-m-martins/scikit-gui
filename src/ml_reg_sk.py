import numpy as np
from sklearn import metrics as me
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.loader import load_arff
from src.shared import RegressionMetrics


def metricas(texto, y_teste, y_pred) -> RegressionMetrics:
    mae = me.mean_absolute_error(y_teste, y_pred)
    mse = me.mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    r2 = me.r2_score(y_teste, y_pred)
    return RegressionMetrics(mae, mse, rmse, r2)


# ==============================================================
# LEITURA DO ARQUIVO
# ==============================================================
def ler_arquivo(arquivo):
    df, _ = load_arff(arquivo)
    atributos = df.iloc[:, :-1].astype(float).values
    target = df.iloc[:, -1].astype(float).values
    return atributos, target


def mlp_treino(x_train, y_train):
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(150,),
        activation="relu",
        solver="lbfgs",
        alpha=0.01,
        learning_rate="adaptive",
        max_iter=1000,
        random_state=42,
    )
    mlp_model.fit(x_train, y_train)
    return mlp_model


def mlp_teste(x_teste, y_teste, mlp_model):
    y_pred = mlp_model.predict(x_teste)
    metricas("MLP", y_teste, y_pred)


def random_forest_treino(x_train, y_train):
    rf_model = RandomForestRegressor(
        n_estimators=80,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    return rf_model


def random_forest_teste(x_teste, y_teste, rf_model):
    y_pred = rf_model.predict(x_teste)
    metricas("RANDOM FOREST", y_teste, y_pred)


# ==============================================================
# MÓDULO PRINCIPAL
# ==============================================================
#   MÓDULO PRINCIPAL[
# ==============================================================
if __name__ == "__main__":
    arquivo = "dados_reg.txt"

    feat, target = ler_arquivo(arquivo)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)

    modelo = mlp_treino(x_treino_sc, y_treino)
    mlp_teste(x_teste_sc, y_teste, modelo)

    modelo = random_forest_treino(x_treino, y_treino)
    random_forest_teste(x_teste, y_teste, modelo)
