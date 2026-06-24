from lark import Lark


gramatica = Lark(r"""
start: comando+

?comando : dataset
         | target
         | problema
         | split
         | scaler
         | pipeline
         | modelo
         | transformador
         | avaliacao
         | crossval

dataset: "dataset" STRING
target: "target" IDENTIFIER
problema: "problem" PROBLEMA
split: "split" param_split ("," param_split)*
scaler: "scaler" SCALER
avaliacao: "evaluate" METRICA
crossval: "crossvalidation" param_cv ("," param_cv)*

modelo: MODELO params_modelo?
transformador: TRANSFORMADOR params_transformador?

pipeline: "pipeline" "{" corpo_pipeline "}"
corpo_pipeline: passo_pipeline ("," passo_pipeline)*
?passo_pipeline: transformador
              | modelo

param_split: TEST_SIZE "=" NUMERO
           | RANDOM_STATE "=" NUMERO

param_cv: CV_FOLDS "=" NUMERO
        | CV_SCORING "=" METRICA

?params_modelo: "{" lista_params "}"
              | "(" lista_params_inline ")"

?params_transformador: "{" lista_params "}"
                     | "(" lista_params_inline ")"

lista_params: param ("," param)*
param: IDENTIFIER "=" valor_param

lista_params_inline: param_inline ("," param_inline)*
param_inline: IDENTIFIER "=" valor_param

?valor_param: NUMERO
            | STRING
            | BOOLEAN
            | IDENTIFIER


// TOKENS:

PROBLEMA: "classification"
         | "regression"

SCALER: "standard"
      | "minmax"

METRICA: "accuracy"
       | "precision"
       | "recall"
       | "f1"
       | "mse"
       | "rmse"
       | "mae"
       | "r2"

MODELO: "RandomForest"
      | "SVM"
      | "LinearRegression"
      | "GradientBoosting"
      | "LogisticRegression"
      | "Ridge"
      | "Lasso"

TRANSFORMADOR: "StandardScaler"
             | "PCA"
             | "SelectKBest"
             | "RobustScaler"
             | "PolynomialFeatures"

TEST_SIZE: "test_size"
RANDOM_STATE: "random_state"
CV_FOLDS: "folds"
CV_SCORING: "scoring"

NUMERO: /-?[0-9]+(\.[0-9]+)?/
BOOLEAN: "true" | "false"
STRING: /"[^"]*"/
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
COMENTARIO: /#.*/
COMENTARIO_BLOCO: /\/\*[\s\S]*?\*\//

%import common.WS
%ignore WS
%ignore COMENTARIO
%ignore COMENTARIO_BLOCO""")
