from enum import Enum


class Methods(str, Enum):
    GAUSSIAN_NAIVE_BAYES = "Gaussian Naive Bayes"
    MULTI_NAIVE_BAYES = "Multinomial Naive Bayes"
    COMPLEMENT_NAIVE_BAYES = "Complement Naive Bayes"
    LOGISTICS = "Logistics"
    LINEAR = "Linear Regression"
    MULTILAYER_PERCEPTRON = "Multilayer Perceptron"
    SVC = "SVC"
    K_NEIGHBORS = "K-Neighbors"
    ADABOOSTM1 = "AdaBoostM1"
    GRADIENT_BOOSTING = "Gradient Boosting"
    HIST_GRADIENT_BOOST = "Hist Gradient Boosting"
    DECISION_TABLE = "Decision table"
    DECISION_TREE = "Decision Tree"
    EXTRA_TREES = "Extra Trees"
    RANDOM_FOREST = "Random Forest"
    RIDGE = "Ridge"
    HUBER = "Huber"
    BAYESIAN_RIDGE = "Bayesian Ridge"

    @staticmethod
    def values():
        return [method.value for method in Methods]


class Modes(str, Enum):
    REGRESSION = "Regressão"
    CLASSIFICATION = "Classificação"

    @staticmethod
    def values():
        return [mode.value for mode in Modes]


class Preprocessors(str, Enum):
    NONE = "Nenhum"
    STANDARD_SCALER = "StandardScaler"
    MIN_MAX_SCALER = "MinMaxScaler"
    ROBUST_SCALER = "RobustScaler"

    @staticmethod
    def values():
        return [preprocessor.value for preprocessor in Preprocessors]


class MethodResultField(str, Enum):
    method_name = "Método:"
    timestamp = "Execução:"
    dataset_name = "Dataset:"
    n_samples = "Amostras:"
    n_features = "Features:"
    n_classes = "Classes:"
    test_split = "Divisão de teste:"
    train_test = "Treino / Teste:"
    load_time = "Tempo de carga:"
    train_time = "Tempo de treino:"
    test_time = "Tempo de teste:"
    total_time = "Tempo total:"
    random_state = "Random state:"
    model_params = "Hiperparâmetros:"
    preprocessing = "Pré-processamento:"
