- [x] R01 Aceitar arquivos no formato arff do weka
- [x] R02 Determinar dataset para regressão
- [x] R03 Adicionar suporte a todos os Métodos Classifier que se adequem do scikit
  - [x] Naive Bayes: Gaussian → GaussianNB, Multinomial → MultinomialNB e Complement → ComplementNB.
  - [x] Lineares e neurais: Logistics → LogisticRegression e Multilayer Perceptron → MLPClassifier.
  - [x] SVM e ensembles:
    - [x] SMO → SVC
    - [x] AdaBoostM1 → AdaBoostClassifier
    - [x] Decision Tree → DecisionTreeClassifier
    - [x] Random Forest → RandomForestClassifier
  - [x] Usar SVC com kernel linear para aproximar o padrão do Weka.
  - Sem equivalente direto: KStar, Logitboost, Decision table e J48.
  - [x] Análogos, não equivalentes:
    - [x] KNeighborsClassifier para KStar
    - [x] GradientBoostingClassifier com log loss para Logitboost
    - [x] DecisionTreeClassifier com criterion entropy para J48
  - Multinomial e Complement exigem features não negativas; SVC, LogisticRegression e MLPClassifier precisam de padronização.
  - Vote exige estimadores base definidos.
  - [x] Ignorar métodos sem suporte em run_all_methods sem interromper a execução.
- [x] R04 Garantir 12 métodos em classificação e regressão, priorizando paridade
  - [x] Integrar ml_reg_sk.py à aplicação.
  - [x] Classificação: 12 métodos já implementados.
  - [x] Regressão: completar os 12 métodos.
  - [x] Compartilhar 8 métodos entre os modos: Multilayer Perceptron, Random Forest, Decision Tree, AdaBoostM1, Gradient Boosting, Hist Gradient Boosting, K-Neighbors.
  - [x] Adicionar os equivalentes de regressão:
    - [x] DecisionTreeRegressor, par de Decision Tree.
  - [x] Corrigir o mapeamento de "Logistics" na regressão: LogisticRegression é classificador; usar LinearRegression.
  - [x] Completar a regressão com os melhores métodos para popularidade no OnlineNewsPopularity:
    - [x] ExtraTreesRegressor.
    - [x] Ridge.
    - [x] HuberRegressor.
  - [x] Manter os 4 exclusivos de classificação sem par direto: Logistics, Gaussian Naive Bayes, Multinomial Naive Bayes e Complement Naive Bayes.
  - [x] Resultado esperado: 12 métodos em cada modo, sendo 8 compartilhados e 4 exclusivos por modo.
  - [x] Usar MAE, MSE, RMSE e R²; padronizar features para modelos sensíveis a escala.
  - [x] Ajustar o scaler apenas no treino e aplicar no teste.
  - [x] Não incluir tuning automático; expor apenas parâmetros relevantes.
  - [x] Validar que run_all_methods executa os 12 métodos sem interrupção.
- [x] R07 Ajustar a configuração para o "Executar Todos" sem grandes gargalos
  - [x] Garantir que a interface continue responsiva e que o contador avance apenas quando cada método concluir.

# Especificações do Professor

Avaliação Parcial 2
Condições de conclusão
Aberto: terça-feira, 3 nov. 2026, 00:00
Vencimento: terça-feira, 10 nov. 2026, 23:59

Considerando:

     Os datasets aplicados à tarefa de classificação e regressão escolhidos, escolhido e utilizando o software WEKA.
    Os algoritmos de classificação e de regressão disponíveis no scikit-learn do Python.

Fazer uma aplicação em Python, contemplando uma interface gráfica contendo os seguintes componentes:

    Botão "Abrir arquivo": deve permitir selecionar um dos datasets e exibir o seu resultado em uma aba denominada "Dados".
    Seletor "Tipo de tarefa": com as opções "Classificação" e "Regressão".
    Seletor "Método": com os respectivos métodos para Classificação e Regressão (quando forem selecionados no seletor anterior).
    Seletor  Padrão: permitindo estar ou não selecionados; se selecionado o algoritmo será executado em configuração padrão, caso contrário, será executado com as configurações contidas em um arquivo texto denominado "CONFIG.TXT"
    Botão "Executar: permite executar o método selecionado e exibir as métricas resultantes na aba "Resultados".
    Aba "Dados": exibir os dados do arquivo após executar o botão "Abrir arquivo".
    Aba "Resultados": exibir as métricas obtidas pelo método executado.

O que entregar:

    Arquivo fonte do programa em Python.
    Datasets no formato ser usado no programa.
    Arquivos CONFIG.txt para cada método.
    Arquivo LeiaMe.txt explicando como executar o programa.
