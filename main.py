from grammar import gramatica
from semantic_analyzer import analiseSemantica
from executor import executar

def compilar(codigo):
    """Compila, valida e executa o codigo DSL"""

    print("\n" + "=" * 70)
    print("COMPILADOR DSL")
    print("=" * 70)
    print("\nCODIGO FONTE:")
    print("-" * 60)
    print(codigo)
    print("-" * 60)

    try:
        # 1. Analise Sintatica
        print("\n[1] Analise Sintatica")
        arvore = gramatica.parse(codigo)
        print("    OK - Sintaxe valida")

        # 2. Analise Semantica
        print("\n[2] Analise Semantica")
        ctx = analiseSemantica(arvore)

        if ctx['warnings']:
            print("\n    WARNINGS:")
            for w in ctx['warnings']:
                print(f"      - [{w['tipo']}] {w['msg']}")
                if w.get('sugestao'):
                    print(f"        SUGESTAO: {w['sugestao']}")

        if ctx['erros']:
            print("\n    ERROS:")
            for e in ctx['erros']:
                print(f"      - [{e['tipo']}] {e['msg']}")
                if e.get('sugestao'):
                    print(f"        SUGESTAO: {e['sugestao']}")

            print("\n" + "=" * 70)
            print("COMPILACAO FALHOU - Erros semanticos encontrados")
            print("=" * 70)
            return None

        print("    OK - Semantica valida")

        # 3. Execucao
        print("\n[3] Execucao")
        resultado = executar(ctx)

        if resultado is None:
            print("\n" + "=" * 70)
            print("EXECUCAO FALHOU")
            print("=" * 70)
            return None

        return resultado

    except Exception as e:
        print(f"\nERRO: {e}")
        return None