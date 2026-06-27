# ProjetoCompiladores

Este projeto implementa um compilador/interpreter para uma DSL voltada a fluxos de Machine Learning. A linguagem permite descrever, em texto, um experimento completo: dataset, alvo, separação treino/teste, pré-processamento, pipeline de transformação/modelo, validação cruzada e métrica de avaliação.

O fluxo geral é este:

1. O código da DSL é analisado sintaticamente com Lark.
2. A árvore gerada passa por validação semântica, que verifica regras como ordem correta das etapas, compatibilidade entre problema e modelo, parâmetros válidos e riscos de vazamento de dados.
3. Se estiver tudo certo, o pipeline é executado com pandas e scikit-learn.

## O que é o projeto

O objetivo é funcionar como um interpretador de DSL para Machine Learning. Em vez de escrever diretamente código Python com scikit-learn, o usuário descreve o experimento em uma linguagem própria, por exemplo:

```text
dataset "data/iris/Iris.csv"
target Species
problem classification
split test_size=0.3, random_state=42
scaler standard
pipeline {
    PCA { n_components=2 },
    RandomForest {
        n_estimators=100,
        max_depth=5,
        random_state=42
    }
}
evaluate accuracy
```

Esse código define um experimento de classificação usando o dataset Iris, aplica divisão treino/teste, normalização, PCA, treino de RandomForest e avaliação por accuracy.

### Comentários

A DSL suporta comentários de linha e de bloco:

```text
# Isso é um comentário de linha

/* Isso é um
   comentário de bloco */
```

## Estrutura do projeto

```text
ProjetoCompiladores/
├── src/
│   ├── __init__.py
│   ├── main.py                # Orquestrador
│   ├── grammar.py             # DSL (Lark)
│   ├── semantic_analyzer.py   # Validação semântica
│   ├── executor.py            # Execução (sklearn)
│   ├── validator.py           # Re-exporta configurações de config.py
│   └── config.py              # Configuração centralizada de modelos, métricas, etc
│
├── data/
│   ├── iris/Iris.csv
│   ├── breast_cancer/breast_cancer.csv
│   ├── diabetes/diabetes.csv
│   ├── digits/digits.csv
│   └── wine/wine.csv
│
├── examples/
│   ├── demo_iris.py           # Classificação — Iris com PCA + RandomForest
│   ├── demo_breast_cancer.py  # Classificação binária — Breast Cancer
│   ├── demo_diabetes.py       # Regressão — Diabetes
│   ├── demo_digits.py         # Classificação multi-classe — Digits
│   ├── demo_wine.py           # Classificação — Wine
│   ├── demo_crossval.py       # Exemplo com cross-validation
│   └── demo_custom.py         # Template para dataset próprio
│
├── tests/
│   ├── test_parser.py
│   ├── test_semantic.py
│   └── test_executor.py
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Como rodar

### 1. Criar e ativar um ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

```powershell
pip install -r requirements.txt
```

### 3. Executar o projeto

```powershell
python examples/demo_iris.py
```

Para testar com testes unitários:

```powershell
pip install -r requirements-dev.txt
pytest tests/
```

Para criar seu próprio exemplo, use `examples/demo_custom.py` como template.

## Referência da linguagem

### Comandos disponíveis

| Comando           | Descrição                                             |
|-------------------|-------------------------------------------------------|
| `dataset`         | Caminho para o arquivo CSV                            |
| `target`          | Nome da coluna alvo                                   |
| `problem`         | Tipo de problema: `classification` ou `regression`    |
| `split`           | Divisão treino/teste (`test_size`, `random_state`)    |
| `scaler`          | Normalização global: `standard` ou `minmax`           |
| `pipeline { }`    | Sequência de transformadores seguidos de um modelo    |
| `crossvalidation` | Validação cruzada (`folds`, `scoring` opcional)       |
| `evaluate`        | Métrica de avaliação final                            |

### Modelos suportados

| Nome DSL             | Tipo                | Classificação | Regressão |
|----------------------|---------------------|:---:|:---:|
| `RandomForest`       | Ensemble            | ✅  | ✅  |
| `SVM`                | Support Vector      | ✅  | ✅  |
| `GradientBoosting`   | Ensemble            | ✅  | ✅  |
| `LogisticRegression` | Linear              | ✅  | ❌  |
| `LinearRegression`   | Linear              | ❌  | ✅  |
| `Ridge`              | Linear regularizado | ❌  | ✅  |
| `Lasso`              | Linear regularizado | ❌  | ✅  |

**Parâmetros aceitos por modelo:**

- **RandomForest:** `n_estimators` (int), `max_depth` (int), `random_state` (int)
- **GradientBoosting:** `n_estimators` (int), `max_depth` (int), `learning_rate` (float), `random_state` (int)
- **SVM:** `C` (float), `kernel` (`linear`, `rbf`, `poly`), `gamma` (`scale`, `auto`)
- **LogisticRegression:** `C` (float), `max_iter` (int), `random_state` (int)
- **LinearRegression:** `fit_intercept` (bool)
- **Ridge:** `alpha` (float), `fit_intercept` (bool)
- **Lasso:** `alpha` (float), `fit_intercept` (bool), `max_iter` (int)

### Transformadores suportados

| Nome DSL             | Parâmetros aceitos                                        |
|----------------------|-----------------------------------------------------------|
| `PCA`                | `n_components` (int), `random_state` (int)                |
| `StandardScaler`     | —                                                         |
| `RobustScaler`       | —                                                         |
| `SelectKBest`        | `k` (int), `score_func` (`f_classif` ou `f_regression`)  |
| `PolynomialFeatures` | `degree` (int, mínimo 2)                                  |

### Métricas disponíveis

- **Classificação:** `accuracy`, `precision`, `recall`, `f1`
- **Regressão:** `mse`, `rmse`, `mae`, `r2`

### Cross-validation

O comando `crossvalidation` pode substituir ou complementar o `split`. Quando usado:

- Classificação usa `StratifiedKFold` automaticamente; cai para `KFold` se alguma classe tiver menos amostras que o número de folds.
- Regressão usa `KFold`.
- Se `split` e `crossvalidation` forem declarados juntos, o `split` é ignorado (warning emitido).

```text
crossvalidation folds=5, scoring=r2
```

## Explicação de cada arquivo

### `src/main.py`

Contém a função `compilar(codigo, verbose=False)`, que coordena todo o processo:

1. Faz a análise sintática;
2. Faz a análise semântica;
3. Exibe erros e warnings;
4. Executa o pipeline se não houver erros;
5. Retorna um dicionário com o resultado enriquecido.

**Campos do resultado retornado:**

| Campo               | Descrição                                              |
|---------------------|--------------------------------------------------------|
| `score`             | Valor da métrica avaliada                              |
| `modelo_nome`       | Nome DSL do modelo utilizado                           |
| `metrica`           | Nome da métrica                                        |
| `n_features`        | Número de features após transformações                 |
| `n_amostras_treino` | Amostras no conjunto de treino                         |
| `n_amostras_teste`  | Amostras no conjunto de teste                          |
| `pipeline_steps`    | Lista de passos executados (`tipo:nome`)               |
| `warnings`          | Avisos semânticos não-bloqueantes                      |
| `duracao_segundos`  | Tempo total de execução                                |
| `cv_scores`         | Lista de scores por fold (apenas em cross-validation)  |
| `std`               | Desvio padrão dos folds (apenas em cross-validation)   |

### `src/grammar.py`

Define a gramática da DSL usando Lark. É aqui que estão as regras aceitas pela linguagem: `dataset`, `target`, `problem`, `split`, `scaler`, `pipeline`, `crossvalidation`, `evaluate`, `modelo` e `transformador`. Também define os tokens válidos, como problemas, modelos, transformadores e métricas.

### `src/semantic_analyzer.py`

Implementa a análise semântica. Depois que a sintaxe é aceita, este arquivo verifica se o código faz sentido do ponto de vista do domínio.

**Erros detectados:**

| Tipo                  | Descrição                                                          |
|-----------------------|--------------------------------------------------------------------|
| `DATASET_INVALIDO`    | Arquivo CSV não encontrado ou malformado                           |
| `TARGET_INVALIDO`     | Coluna alvo não existe no dataset                                  |
| `MODELO_INCOMPATIVEL` | Modelo não suporta o tipo de problema declarado                    |
| `METRICA_INVALIDA`    | Métrica incompatível com o tipo de problema                        |
| `VAZAMENTO_DADOS`     | Transformador ou scaler aplicado antes do split                    |
| `PARAMETRO_INVALIDO`  | Parâmetro desconhecido, tipo errado ou fora do intervalo válido    |
| `CONTEXTO_INVALIDO`   | Modelo ou transformador direto quando pipeline já existe           |
| `PIPELINE_VAZIO`      | Pipeline declarado sem nenhum passo                                |
| `PIPELINE_INCOMPLETO` | Pipeline termina com transformador em vez de modelo                |
| `DEPENDENCIA_AUSENTE` | Comando dependente de outro não declarado antes                    |
| `REDUNDANCIA`         | `scaler standard` + `StandardScaler` no pipeline (dupla normalização) |
| `SEM_MODELO`          | Nenhum modelo definido no código                                   |

**Warnings** (não bloqueiam a execução):

| Tipo            | Descrição                                          |
|-----------------|----------------------------------------------------|
| `REDUNDANCIA`   | `split` e `crossvalidation` declarados juntos      |
| `SEM_AVALIACAO` | Nenhuma métrica definida com `evaluate`            |

### `src/executor.py`

Executa o pipeline já validado usando pandas e scikit-learn: carrega o CSV, separa features e target, executa `train_test_split`, aplica scaler, executa os passos do pipeline, treina o modelo e calcula a métrica. Suporta também execução via cross-validation quando `crossvalidation` é declarado.

### `src/config.py`

Arquivo centralizado com configuração de modelos, transformadores, métricas e escaladores.

**Para adicionar um novo modelo, edite apenas este arquivo:**

1. Importe a classe sklearn no topo:
   ```python
   from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
   ```
2. Adicione ao `MAPA_MODELOS_SKLEARN`:
   ```python
   'GradientBoosting': {
       'classification': GradientBoostingClassifier,
       'regression': GradientBoostingRegressor,
   }
   ```
3. Adicione ao `MAPA_MODELOS` (mapa de strings para validação semântica):
   ```python
   'GradientBoosting': {
       'classification': 'GradientBoostingClassifier',
       'regression': 'GradientBoostingRegressor',
   }
   ```
4. Adicione aos `PARAMS_VALIDOS`:
   ```python
   'GradientBoosting': {
       'n_estimators': {'tipo': int, 'min': 1},
       'learning_rate': {'tipo': float, 'min': 0},
       'max_depth': {'tipo': int, 'min': 1},
       'random_state': {'tipo': int},
   }
   ```
5. Adicione o token `MODELO` em `src/grammar.py`:
   ```
   MODELO: "RandomForest"
         | "SVM"
         | "GradientBoosting"   ← aqui
         | ...
   ```

### `src/validator.py`

Re-exporta as configurações de `config.py`. Mantém a separação de responsabilidades sem duplicar código.

### `examples/`

Scripts de demonstração prontos para rodar:

```bash
python examples/demo_iris.py           # Classificação — Iris
python examples/demo_breast_cancer.py  # Classificação binária — Breast Cancer
python examples/demo_diabetes.py       # Regressão — Diabetes
python examples/demo_digits.py         # Classificação multi-classe — Digits
python examples/demo_wine.py           # Classificação — Wine
python examples/demo_crossval.py       # Cross-validation
python examples/demo_custom.py         # Template para dataset próprio
```

### `tests/`

Testes unitários usando pytest:
- `test_parser.py`: valida se o parser DSL funciona
- `test_semantic.py`: valida análise semântica
- `test_executor.py`: valida execução do pipeline

## Como adicionar novos datasets

1. Crie uma pasta em `data/seu_dataset/`
2. Coloque seu CSV lá (ex: `data/seu_dataset/dados.csv`)
3. Referencie na DSL:
   ```text
   dataset "data/seu_dataset/dados.csv"
   target sua_coluna_alvo
   problem classification
   ```

O caminho pode ser absoluto, relativo ao diretório de execução, ou relativo à raiz do projeto — o analisador semântico tenta as três formas automaticamente.

## Observações importantes

- A **ordem das instruções importa**: `dataset` antes de `target`, `split` antes de transformadores e modelos.
- **Data leakage** é detectado automaticamente: transformadores como `PCA`, `StandardScaler`, `SelectKBest`, `RobustScaler` e `PolynomialFeatures` só podem ser usados dentro do pipeline, após o `split`.
- O pipeline **deve terminar com um modelo**; terminar com transformador é erro semântico.
- `crossvalidation` dispensa o `split` — mas declará-los juntos apenas emite um warning, não um erro.
