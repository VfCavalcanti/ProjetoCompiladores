# ProjetoCompiladores

Este projeto implementa um pequeno compilador/interpreter para uma DSL voltada a fluxos de Machine Learning. A linguagem permite descrever, em texto, um experimento com dataset, alvo, separação treino/teste, pré-processamento, pipeline de transformação/modelo e métrica de avaliação.

O fluxo geral é este:

1. O código da DSL é analisado sintaticamente com Lark.
2. A árvore gerada passa por validação semântica, que verifica regras como ordem correta das etapas, compatibilidade entre problema e modelo, parâmetros válidos e riscos de vazamento de dados.
3. Se estiver tudo certo, o pipeline é executado com pandas e scikit-learn.

## O que é o projeto

O objetivo é funcionar como um interpretador de DSL para Machine Learning. Em vez de escrever diretamente código Python com scikit-learn, o usuário descreve o experimento em uma linguagem própria, por exemplo:

```text
dataset "Iris.csv"
target Species
problem classification
split test_size=0.3, random_state=42
scaler standard
pipeline {
    StandardScaler,
    PCA {
        n_components=2
    },
    RandomForest {
        n_estimators=100,
        max_depth=5,
        random_state=42
    }
}
evaluate accuracy
```

Esse código define um experimento de classificação usando o dataset Iris, aplica divisão treino/teste, normalização, PCA, treino de RandomForest e avaliação por accuracy.

## Estrutura do projeto

O projeto agora está bem organizado para facilitar extensões:

```text
ProjetoCompiladores/
├── src/
│   ├── __init__.py
│   ├── main.py                # Orquestrador
│   ├── grammar.py             # DSL (Lark)
│   ├── semantic_analyzer.py   # Validação semântica
│   ├── executor.py            # Execução (sklearn)
│   └── config.py              # Configuração centralizada de modelos, métricas, etc
│
├── data/
│   ├── iris/                  # Dataset Iris
│   │   └── Iris.csv
│   └── (adicione outros datasets aqui)
│
├── examples/                  # Scripts de demonstração
│   ├── demo_iris.py          # Exemplo com Iris
│   └── demo_custom.py        # Template para outro dataset
│
├── tests/                     # Testes unitários
│   ├── test_parser.py        # Testa parser DSL
│   ├── test_semantic.py      # Testa validação semântica
│   └── test_executor.py      # Testa execução
│
├── requirements.txt           # Dependências
├── requirements-dev.txt       # Dependências de desenvolvimento (pytest)
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

O ponto de entrada é o arquivo `examples/demo_iris.py`:

```powershell
python examples/demo_iris.py
```

Para testar com testes unitários:

```powershell
pip install -r requirements-dev.txt
pytest tests/
```

Para criar seu próprio exemplo, use `examples/demo_custom.py` como template.

## Explicação de cada arquivo

### `src/main.py`

Contém a função `compilar(codigo)`, que coordena todo o processo:

1. imprime o código-fonte recebido;
2. faz a análise sintática;
3. faz a análise semântica;
4. exibe erros e warnings;
5. executa o pipeline se não houver erros.

É o núcleo de orquestração do projeto.

Esse é o ponto de entrada lógico do compilador.

### `src/grammar.py`

Define a gramática da DSL usando Lark. É aqui que estão as regras aceitas pela linguagem, como:

- `dataset`
- `target`
- `problem`
- `split`
- `scaler`
- `pipeline`
- `evaluate`
- `modelo`
- `transformador`

Também define os tokens válidos, como problemas (`classification`, `regression`), modelos (`RandomForest`, `SVM`, `LinearRegression`) e métricas.

Esse arquivo concentra a definição da linguagem.

### `src/semantic_analyzer.py`

Implementa a análise semântica. Depois que a sintaxe é aceita, este arquivo verifica se o código faz sentido do ponto de vista do domínio.

Ele valida, por exemplo:

- se o dataset foi definido antes do `target`;
- se o target existe nas colunas do CSV;
- se o tipo de problema é válido;
- se o `split` foi informado antes de etapas sensíveis;
- se há risco de vazamento de dados ao aplicar transformadores antes da divisão;
- se os parâmetros dos modelos e transformadores são válidos;
- se a métrica é compatível com classification ou regression;
- se o pipeline termina com um modelo.

Ao final, retorna um contexto com dados prontos para execução, além de erros e warnings.

Esse arquivo valida as regras de domínio da DSL.

### `src/executor.py`

Executa o pipeline já validado usando pandas e scikit-learn.

O arquivo faz as seguintes etapas:

- carrega o CSV;
- separa features e target;
- executa `train_test_split`;
- aplica scaler, se houver;
- executa os passos do pipeline;
- treina o modelo final;
- gera predições;
- calcula a métrica final.

Se algo falhar durante o processamento, a execução é interrompida e o erro é exibido.

Esse arquivo executa o pipeline validado.

### `src/config.py`

Arquivo centralizado com configuração de modelos, transformadores, métricas e escaladores. 

Contém:
- Mapas de modelos sklearn e seus parâmetros válidos
- Transformadores disponíveis
- Escaladores (standard, minmax)
- Métricas válidas por tipo de problema
- Transformadores que causam vazamento de dados

**Para adicionar novo modelo, simplesmente edite este arquivo e adicione:**
1. Importe a classe sklearn no topo
2. Adicione ao dicionário MAPA_MODELOS_SKLEARN
3. Adicione aos PARAMS_VALIDOS
4. Pronto!

### `src/validator.py`

Simplificado para importar configurações de `config.py`. Mantém a lógica de validação.

### `examples/demo_iris.py`

Script de demonstração com dataset Iris (classificação com RandomForest + PCA).

### `examples/demo_custom.py`

Template para criar seu próprio exemplo com outro dataset.

### `tests/`

Testes unitários usando pytest:
- `test_parser.py`: valida se o parser DSL funciona
- `test_semantic.py`: valida análise semântica
- `test_executor.py`: valida execução do pipeline

## Observações importantes

- O projeto evita vazamento de dados: a ordem das instruções na DSL importa
- Use `examples/demo_iris.py` como ponto de partida
- A validação semântica é crucial: código válido sintaticamente pode ser rejeitado por regras de domínio

## Como adicionar novos modelos

1. Abra `src/config.py`
2. Importe as classes sklearn no topo:
   ```python
   from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
   ```
3. Adicione ao `MAPA_MODELOS_SKLEARN`:
   ```python
   'GradientBoosting': {
       'classification': GradientBoostingClassifier,
       'regression': GradientBoostingRegressor,
   }
   ```
4. Adicione os parâmetros válidos em `PARAMS_VALIDOS`:
   ```python
   'GradientBoosting': {
       'n_estimators': {'tipo': int, 'min': 1},
       'learning_rate': {'tipo': float, 'min': 0},
       'max_depth': {'tipo': int, 'min': 1},
   }
   ```
5. Use na DSL:
   ```
   pipeline {
       GradientBoosting {
           n_estimators=100,
           learning_rate=0.1,
           max_depth=5
       }
   }
   ```

## Como adicionar novos datasets

1. Crie uma pasta em `data/seu_dataset/`
2. Coloque seu CSV lá (ex: `data/seu_dataset/dados.csv`)
3. Use em `examples/demo_custom.py` ou crie um novo script
4. Referencie na DSL:
   ```
   dataset "data/seu_dataset/dados.csv"
   target sua_coluna_alvo
   problem classification
   ```

## Como rodar testes

```bash
pip install -r requirements-dev.txt
pytest tests/
```