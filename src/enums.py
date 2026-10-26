from enum import Enum


class Methods(str, Enum):
    GAUSSIAN_NAIVE_BAYES = "Gaussian Naive Bayes"
    MULTI_NAIVE_BAYES = "Multinomial Naive Bayes"
    COMPLEMENT_NAIVE_BAYES = "Complement Naive Bayes"
    LOGISTICS = "Logistics"
    MULTILAYER_PERCEPTRON = "Multilayer Perceptron"
    SMO = "SMO"
    K_STAR = "KStar"
    ADABOOSTM1 = "AdaBoostM1"
    LOGITBOOST = "Logitboost"
    VOTE = "Vote"
    DECISION_TABLE = "Decision table"
    PART = "PART"
    J48 = "J48"
    RANDOM_FOREST = "Random Forest"

    @staticmethod
    def values():
        return [method.value for method in Methods]
