import time
import lark.exceptions
from src.grammar import gramatica
from src.semantic_analyzer import analiseSemantica
from src.executor import executar


def compilar(codigo, verbose=False):
    """Compila, valida e executa o codigo DSL.

    Args:
        codigo: String com código DSL
        verbose: Se True, exibe prints detalhados. Se False, roda silenciosamente.

    Returns:
        Dict com resultado enriquecido ou None se falhar
    """
    inicio = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("COMPILADOR DSL")
        print("=" * 70)
        print("\nCODIGO FONTE:")
        print("-" * 60)
        print(codigo)
        print("-" * 60)

    try:
        if verbose:
            print("\n[1] Analise Sintatica")
        try:
            arvore = gramatica.parse(codigo)
        except lark.exceptions.UnexpectedInput as e:
            contexto = e.get_context(codigo)
            msg = (
                f"Erro sintático na linha {e.line}, coluna {e.column}:\n"
                f"{contexto}"
            )
            if verbose:
                print(f"    ERRO SINTATICO:\n{msg}")
            raise SyntaxError(msg) from e

        if verbose:
            print("    OK - Sintaxe valida")

        if verbose:
            print("\n[2] Analise Semantica")
        ctx = analiseSemantica(arvore)

        if ctx['warnings']:
            if verbose:
                print("\n    WARNINGS:")
                for w in ctx['warnings']:
                    print(f"      - [{w['tipo']}] {w['msg']}")
                    if w.get('sugestao'):
                        print(f"        SUGESTAO: {w['sugestao']}")

        if ctx['erros']:
            if verbose:
                print("\n    ERROS:")
                for e in ctx['erros']:
                    print(f"      - [{e['tipo']}] {e['msg']}")
                    if e.get('sugestao'):
                        print(f"        SUGESTAO: {e['sugestao']}")
                print("\n" + "=" * 70)
                print("COMPILACAO FALHOU - Erros semanticos encontrados")
                print("=" * 70)
            return None

        if verbose:
            print("    OK - Semantica valida")

        if verbose:
            print("\n[3] Execucao")
        resultado = executar(ctx)

        if resultado is None:
            if verbose:
                print("\n" + "=" * 70)
                print("EXECUCAO FALHOU")
                print("=" * 70)
            return None

        duracao = time.time() - inicio

        # Enriquece o resultado com metadados
        modelo_nome = None
        pipeline_steps = []
        for passo in ctx.get('pipeline', []):
            pipeline_steps.append(f"{passo['tipo']}:{passo['nome']}")
            if passo['tipo'] == 'modelo':
                modelo_nome = passo['nome']

        n_features = None
        n_amostras_treino = None
        n_amostras_teste = None

        if 'X_treino' in resultado:
            X_treino = resultado['X_treino']
            X_teste = resultado['X_teste']
            import numpy as np
            if hasattr(X_treino, 'shape'):
                n_features = X_treino.shape[1] if len(X_treino.shape) > 1 else 1
                n_amostras_treino = X_treino.shape[0]
                n_amostras_teste = X_teste.shape[0]

        resultado.update({
            'metrica': ctx.get('metrica'),
            'modelo_nome': modelo_nome,
            'n_features': n_features,
            'n_amostras_treino': n_amostras_treino,
            'n_amostras_teste': n_amostras_teste,
            'pipeline_steps': pipeline_steps,
            'warnings': ctx.get('warnings', []),
            'duracao_segundos': round(duracao, 4),
        })

        return resultado

    except SyntaxError:
        raise
    except Exception as e:
        if verbose:
            print(f"\nERRO: {e}")
        return None
