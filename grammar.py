from lark import Lark


gramatica = Lark("""
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

dataset: "dataset" STRING
target: "target" IDENTIFIER
problema: "problem" PROBLEMA
split: "split" param_split ("," param_split)*
scaler: "scaler" SCALER
avaliacao: "evaluate" METRICA

modelo: MODELO params_modelo?
transformador: TRANSFORMADOR params_transformador?

pipeline: "pipeline" "{" corpo_pipeline "}"
corpo_pipeline: passo_pipeline ("," passo_pipeline)*
?passo_pipeline: transformador
              | modelo

param_split: TEST_SIZE "=" NUMERO
           | RANDOM_STATE "=" NUMERO

// CORRIGIDO: inlinados, sem ramo intermediário inútil
?params_modelo: "{" lista_params "}"
              | "(" lista_params_inline ")"

?params_transformador: "{" lista_params "}"
                     | "(" lista_params_inline ")"

lista_params: param ("," param)*
param: IDENTIFIER "=" valor_param

lista_params_inline: param_inline ("," param_inline)*
param_inline: IDENTIFIER "=" valor_param

// CORRIGIDO: inlinado, sem ramo intermediário inútil
?valor_param: NUMERO
            | STRING
            | BOOLEAN
            | IDENTIFIER

// =========================
// TOKENS
// =========================

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

TRANSFORMADOR: "StandardScaler"
             | "PCA"
             | "SelectKBest"

TEST_SIZE: "test_size"
RANDOM_STATE: "random_state"

NUMERO: /-?[0-9]+(\.[0-9]+)?/
BOOLEAN: "true" | "false"
STRING: /"[^"]*"/
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
COMENTARIO: /#.*/

%import common.WS
%ignore WS
%ignore COMENTARIO""")