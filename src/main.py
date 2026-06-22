from src.grammar import gramatica
from src.semantic_analyzer import analiseSemantica
from src.executor import executar


def compilar(codigo, verbose=False):
    """Compila, valida e executa o codigo DSL.

    Args:
        codigo: String com código DSL
        verbose: Se True, exibe prints detalhados. Se False, roda silenciosamente.

    Returns:
        Dict com resultado ou None se falhar
    """
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
        arvore = gramatica.parse(codigo)
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

        return resultado

    except Exception as e:
        if verbose:
            print(f"\nERRO: {e}")
        return None
