from src.enums import Methods


def test_methods_values_order():
    assert Methods.values() == [
        "Naive Bayes",
        "Logistics",
        "Multilayer Perceptron",
        "SMO",
        "KStar",
        "AdaBoostM1",
        "Logitboost",
        "Vote",
        "Decision table",
        "PART",
        "J48",
        "Random Forest",
    ]
