import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from validator import (
    mapa_modelos_sklearn,
    mapa_transformadores_sklearn,
    mapa_scalers_sklearn,
    mapa_metricas_sklearn
)

def executar(ctx):
    """Executa o pipeline usando sklearn"""

    print("\n" + "=" * 70)
    print("EXECUTANDO PIPELINE")
    print("=" * 70)

    # 1. Carrega dataset
    print(f"\n[1] Carregando dataset: {ctx['dataset']}")

    try:
        df = pd.read_csv(ctx['dataset'])
        print(f"    Shape: {df.shape}")
        print(f"    Colunas: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"    ERRO ao carregar dataset: {e}")
        return None

    # 2. Separa features e target
    print(f"\n[2] Separando features e target")
    target = ctx['target']

    if target not in df.columns:
        print(f"    ERRO: Target '{target}' nao encontrado no dataset")
        return None

    X = df.drop(target, axis=1)
    y = df[target]

    print(f"    Target: {target}")
    print(f"    Features: {', '.join(X.columns.tolist())}")
    print(f"    Amostras: {len(X)}")

    # 3. Split treino/teste
    print(f"\n[3] Dividindo dados em treino e teste")

    if not ctx['split']:
        print("    ERRO: Nenhum split definido")
        return None

    test_size = ctx['split'].get('test_size', 0.2)
    random_state = ctx['split'].get('random_state', 42)

    print(f"    test_size: {test_size}")
    print(f"    random_state: {random_state}")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if ctx['problema'] == 'classification' else None
    )

    print(f"    Treino: {len(X_treino)} amostras")
    print(f"    Teste: {len(X_teste)} amostras")

    # 4. Aplica Scaler (se definido)
    scaler_nome = ctx.get('scaler')
    scaler_obj = None

    if scaler_nome:
        print(f"\n[4] Aplicando Scaler: {scaler_nome}")
        scaler_class = mapa_scalers_sklearn.get(scaler_nome)

        if scaler_class:
            scaler_obj = scaler_class()
            X_treino = scaler_obj.fit_transform(X_treino)
            X_teste = scaler_obj.transform(X_teste)
            print(f"    Scaler aplicado com sucesso")
        else:
            print(f"    ERRO: Scaler '{scaler_nome}' nao encontrado")
            return None

    # 5. Executa pipeline
    print(f"\n[5] Executando pipeline")

    pipeline_steps = ctx['pipeline']
    modelo_final = None
    X_treino_atual = X_treino
    X_teste_atual = X_teste

    for i, passo in enumerate(pipeline_steps, 1):
        tipo = passo['tipo']
        nome = passo['nome']
        params = passo['params']

        print(f"    Passo {i}: {nome} ({tipo})")

        if tipo == 'transformador':
            transformador_class = mapa_transformadores_sklearn.get(nome)

            if not transformador_class:
                print(f"        ERRO: Transformador '{nome}' nao encontrado")
                return None

            try:
                transformador = transformador_class(**params)
                X_treino_atual = transformador.fit_transform(X_treino_atual)
                X_teste_atual = transformador.transform(X_teste_atual)
                print(f"        Transformador aplicado")
                if params:
                    print(f"        Parametros: {params}")
            except Exception as e:
                print(f"        ERRO ao aplicar transformador: {e}")
                return None

        elif tipo == 'modelo':
            problema = ctx['problema']
            model_class = mapa_modelos_sklearn.get(nome, {}).get(problema)

            if not model_class:
                print(f"        ERRO: Modelo '{nome}' nao suporta {problema}")
                return None

            try:
                modelo = model_class(**params)
                print(f"        Treinando modelo...")
                modelo.fit(X_treino_atual, y_treino)
                modelo_final = modelo
                print(f"        Modelo treinado com sucesso")
                if params:
                    print(f"        Parametros: {params}")
            except Exception as e:
                print(f"        ERRO ao treinar modelo: {e}")
                return None

    # 6. Faz predicoes
    print(f"\n[6] Fazendo predicoes")

    if modelo_final is None:
        print("    ERRO: Nenhum modelo foi treinado")
        return None

    try:
        y_pred = modelo_final.predict(X_teste_atual)
        print(f"    Predicoes feitas para {len(y_pred)} amostras")
    except Exception as e:
        print(f"    ERRO ao fazer predicoes: {e}")
        return None

    # 7. Avaliacao
    print(f"\n[7] Avaliando modelo")

    metrica = ctx['metrica']

    if not metrica:
        print("    WARNING: Nenhuma metrica definida para avaliacao")
        return {
            'modelo': modelo_final,
            'X_treino': X_treino_atual,
            'X_teste': X_teste_atual,
            'y_treino': y_treino,
            'y_teste': y_teste,
            'y_pred': y_pred,
            'score': None
        }

    func_metrica = mapa_metricas_sklearn.get(ctx['problema'], {}).get(metrica)

    if not func_metrica:
        print(f"    ERRO: Metrica '{metrica}' nao encontrada para {ctx['problema']}")
        return None

    try:
        if ctx['problema'] == 'classification':
            if isinstance(y_teste.iloc[0], str):
                le = LabelEncoder()
                y_teste_encoded = le.fit_transform(y_teste)
                y_pred_encoded = le.transform(y_pred)
                score = func_metrica(y_teste_encoded, y_pred_encoded)
            else:
                score = func_metrica(y_teste, y_pred)
        else:
            score = func_metrica(y_teste, y_pred)

        print(f"    {metrica}: {score:.4f}")

    except Exception as e:
        print(f"    ERRO ao calcular metrica: {e}")
        return None

    print("\n" + "=" * 70)
    print("EXECUCAO CONCLUIDA COM SUCESSO")
    print("=" * 70)
    print(f"\nRESULTADO FINAL:")
    print(f"  Modelo: {type(modelo_final).__name__}")
    print(f"  Metrica: {metrica} = {score:.4f}")

    return {
        'modelo': modelo_final,
        'X_treino': X_treino_atual,
        'X_teste': X_teste_atual,
        'y_treino': y_treino,
        'y_teste': y_teste,
        'y_pred': y_pred,
        'score': score
    }