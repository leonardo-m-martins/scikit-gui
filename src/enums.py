from enum import Enum

class Methods(str, Enum):
  NAIVE_BAYES = "Naive Bayes"
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