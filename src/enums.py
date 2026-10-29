from enum import Enum


class Methods(str, Enum):
    GAUSSIAN_NAIVE_BAYES = "Gaussian Naive Bayes"
    MULTI_NAIVE_BAYES = "Multinomial Naive Bayes"
    COMPLEMENT_NAIVE_BAYES = "Complement Naive Bayes"
    LOGISTICS = "Logistics"
    MULTILAYER_PERCEPTRON = "Multilayer Perceptron"
    SVC = "SVC"
    K_STAR = "KStar"
    ADABOOSTM1 = "AdaBoostM1"
    LOGITBOOST = "Logitboost"
    VOTE = "Vote"
    DECISION_TABLE = "Decision table"
    DECISION_TREE = "Decision Tree"
    J48 = "J48"
    RANDOM_FOREST = "Random Forest"

    @staticmethod
    def values():
        return [method.value for method in Methods]


class Modes(str, Enum):
    REGRESSION = "Regressão"
    CLASSIFICATION = "Classificação"

    @staticmethod
    def values():
        return [mode.value for mode in Modes]


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
