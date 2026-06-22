from lark import Token, Tree
import pandas as pd
from src.validator import (
    mapa_modelos,
    transformadores_com_vazamento,
    metricas_validas,
    params_validos
)

def analiseSemantica(tree):
    ctx = {
        'dataset': None,
        'colunas': set(),
        'target': None,
        'problema': None,
        'split': {},
        'split_feito': False,
        'scaler': None,
        'pipeline': [],
        'modelo_direto': None,
        'transformador_direto': None,
        'metrica': None,
        'erros': [],
        'warnings': []
    }

    def extrair_valor(no):
        if isinstance(no, Token):
            if no.type == 'STRING':
                return no.value[1:-1]

            elif no.type == 'NUMERO':
                if '.' in no.value:
                    return float(no.value)
                return int(no.value)

            elif no.type == 'BOOLEAN':
                return no.value.lower() == 'true'

            return no.value

        elif isinstance(no, Tree):
            if len(no.children) == 1:
                return extrair_valor(no.children[0])

        return no
    def extrair_nome(no):
        """Extrai o nome de um no que pode ser Token ou Tree"""
        if isinstance(no, Token):
            return str(no)
        elif isinstance(no, Tree):
            # Se for uma regra como tipo_problema, pega o primeiro filho
            if no.children:
                return extrair_nome(no.children[0])
            return str(no)
        return str(no)

    def extrair_params(no):
        params = {}

        if not isinstance(no, Tree):
            return params

        match no.data:
            case 'lista_params':
                for filho in no.children:
                    if isinstance(filho, Tree) and filho.data == 'param':
                        chave = str(filho.children[0])
                        valor = extrair_valor(filho.children[1])
                        params[chave] = valor

            case 'lista_params_inline':
                for filho in no.children:
                    if isinstance(filho, Tree) and filho.data == 'param_inline':
                        chave = str(filho.children[0])
                        valor = extrair_valor(filho.children[1])
                        params[chave] = valor

            case 'params_modelo' | 'params_transformador':
                for filho in no.children:
                    if isinstance(filho, Tree):
                        params.update(extrair_params(filho))

        return params

    def validar_params(nome, params):
        if nome not in params_validos:
            return

        for param, valor in params.items():
            if param not in params_validos[nome]:
                ctx['erros'].append({
                    'tipo': 'PARAMETRO_INVALIDO',
                    'msg': f"Parâmetro desconhecido '{param}' para {nome}",
                    'sugestao': f"Válidos: {', '.join(params_validos[nome].keys())}"
                })
            else:
                spec = params_validos[nome][param]
                if 'min' in spec and valor < spec['min']:
                    ctx['erros'].append({
                        'tipo': 'PARAMETRO_INVALIDO',
                        'msg': f"'{param}' deve ser >= {spec['min']}, recebido: {valor}"
                    })
                if 'opcoes' in spec and valor not in spec['opcoes']:
                    ctx['erros'].append({
                        'tipo': 'PARAMETRO_INVALIDO',
                        'msg': f"'{param}' deve ser {spec['opcoes']}, recebido: {valor}"
                    })

    def percorrer(no):
        if isinstance(no, Token):
            return

        if not isinstance(no, Tree):
            if hasattr(no, 'children'):
                for filho in no.children:
                    percorrer(filho)
            return

        match no.data:

            case 'dataset':
                caminho = no.children[0].value[1:-1]
                ctx['dataset'] = caminho

                try:
                    df = pd.read_csv(caminho)
                    ctx['colunas'] = set(df.columns)
                except:
                    ctx['colunas'] = {
                        'Species', 'SepalLengthCm', 'SepalWidthCm',
                        'PetalLengthCm', 'PetalWidthCm'
                    }

            case 'target':
                target = extrair_nome(no.children[0])

                if not ctx['dataset']:
                    ctx['erros'].append({
                        'tipo': 'DEPENDENCIA_AUSENTE',
                        'msg': 'Dataset deve ser definido antes do target',
                        'sugestao': 'Defina "dataset" antes de "target"'
                    })
                elif target not in ctx['colunas']:
                    ctx['erros'].append({
                        'tipo': 'TARGET_INVALIDO',
                        'msg': f"Target '{target}' não encontrado no dataset",
                        'sugestao': f"Colunas disponíveis: {', '.join(sorted(ctx['colunas']))}"
                    })
                else:
                    ctx['target'] = target

            case 'problema':
                problema = extrair_nome(no.children[0])

                if problema not in ['classification', 'regression']:
                    ctx['erros'].append({
                        'tipo': 'PROBLEMA_INVALIDO',
                        'msg': f"Problema inválido: '{problema}'",
                        'sugestao': 'Use "classification" ou "regression"'
                    })
                else:
                    ctx['problema'] = problema

            case 'split':
                params = {}

                for filho in no.children:
                    if isinstance(filho, Tree) and filho.data == 'param_split':
                        chave = str(filho.children[0])
                        valor = filho.children[1]
                        if isinstance(valor, Token):
                            if '.' in valor.value:
                                valor = float(valor.value)
                            else:
                                valor = int(valor.value)
                        params[chave] = valor

                if 'test_size' in params:
                    if not (0 < params['test_size'] < 1):
                        ctx['erros'].append({
                            'tipo': 'PARAMETRO_INVALIDO',
                            'msg': f"test_size deve estar entre 0 e 1, recebido: {params['test_size']}"
                        })

                if 'random_state' in params:
                    if not isinstance(params['random_state'], int) or params['random_state'] < 0:
                        ctx['erros'].append({
                            'tipo': 'PARAMETRO_INVALIDO',
                            'msg': f"random_state deve ser >= 0, recebido: {params['random_state']}"
                        })

                ctx['split'] = params
                ctx['split_feito'] = True

            case 'scaler':
                scaler = extrair_nome(no.children[0])

                if scaler not in ['standard', 'minmax']:
                    ctx['erros'].append({
                        'tipo': 'PARAMETRO_INVALIDO',
                        'msg': f"Scaler inválido: '{scaler}'",
                        'sugestao': 'Use "standard" ou "minmax"'
                    })

                if not ctx['split_feito']:
                    ctx['erros'].append({
                        'tipo': 'VAZAMENTO_DADOS',
                        'msg': f"Scaler '{scaler}' aplicado ANTES do split",
                        'sugestao': 'Mova "scaler" para depois de "split"'
                    })

                ctx['scaler'] = scaler

            case 'transformador':
                nome = extrair_nome(no.children[0])
                params = {}

                if len(no.children) > 1:
                    params = extrair_params(no.children[1])

                if ctx['pipeline']:
                    ctx['erros'].append({
                        'tipo': 'CONTEXTO_INVALIDO',
                        'msg': f"Transformador '{nome}' fora do pipeline, mas pipeline já existe",
                        'sugestao': 'Coloque o transformador dentro do pipeline'
                    })

                if ctx['transformador_direto']:
                    ctx['erros'].append({
                        'tipo': 'CONTEXTO_INVALIDO',
                        'msg': f"Múltiplos transformadores diretos: '{nome}' e '{ctx['transformador_direto']}'",
                        'sugestao': 'Use pipeline para múltiplos passos'
                    })

                if nome in transformadores_com_vazamento:
                    if not ctx['split_feito']:
                        ctx['erros'].append({
                            'tipo': 'VAZAMENTO_DADOS',
                            'msg': f"Transformador '{nome}' ANTES do split",
                            'sugestao': f'Defina "split" antes de "{nome}"'
                        })

                validar_params(nome, params)

                ctx['transformador_direto'] = nome
                ctx['pipeline'].append({
                    'tipo': 'transformador',
                    'nome': nome,
                    'params': params
                })

            case 'modelo':
                nome = extrair_nome(no.children[0])
                params = {}

                if len(no.children) > 1:
                    params = extrair_params(no.children[1])

                if ctx['pipeline']:
                    ctx['erros'].append({
                        'tipo': 'CONTEXTO_INVALIDO',
                        'msg': f"Modelo '{nome}' fora do pipeline, mas pipeline já existe",
                        'sugestao': 'Coloque o modelo dentro do pipeline ou remova o pipeline'
                    })

                if ctx['modelo_direto']:
                    ctx['erros'].append({
                        'tipo': 'CONTEXTO_INVALIDO',
                        'msg': f"Múltiplos modelos diretos: '{nome}' e '{ctx['modelo_direto']}'",
                        'sugestao': 'Use pipeline para múltiplos passos'
                    })

                if not ctx['split_feito']:
                    ctx['erros'].append({
                        'tipo': 'VAZAMENTO_DADOS',
                        'msg': f"Modelo '{nome}' sem split definido",
                        'sugestao': 'Defina "split" antes do modelo'
                    })

                validar_params(nome, params)

                if ctx['problema']:
                    if nome in mapa_modelos:
                        if ctx['problema'] not in mapa_modelos[nome]:
                            ctx['erros'].append({
                                'tipo': 'MODELO_INCOMPATIVEL',
                                'msg': f"'{nome}' não suporta {ctx['problema']}",
                                'sugestao': f"Use um modelo compatível com {ctx['problema']}"
                            })

                ctx['modelo_direto'] = nome
                ctx['pipeline'].append({
                    'tipo': 'modelo',
                    'nome': nome,
                    'params': params
                })

            case 'pipeline':
                if ctx['modelo_direto'] or ctx['transformador_direto']:
                    ctx['erros'].append({
                        'tipo': 'CONTEXTO_INVALIDO',
                        'msg': 'Pipeline declarado, mas já existem declarações diretas',
                        'sugestao': 'Use apenas pipeline OU declarações diretas'
                    })

                if not ctx['problema']:
                    ctx['erros'].append({
                        'tipo': 'DEPENDENCIA_AUSENTE',
                        'msg': 'Problema deve ser definido antes do pipeline',
                        'sugestao': 'Defina "problem" antes de "pipeline"'
                    })

                for filho in no.children:
                    if isinstance(filho, Tree) and filho.data == 'corpo_pipeline':
                        for passo in filho.children:
                            if isinstance(passo, Tree):
                                if passo.data == 'transformador':
                                    nome = extrair_nome(passo.children[0])
                                    params = {}

                                    if len(passo.children) > 1:
                                        params = extrair_params(passo.children[1])

                                    if nome in transformadores_com_vazamento:
                                        if not ctx['split_feito']:
                                            ctx['erros'].append({
                                                'tipo': 'VAZAMENTO_DADOS',
                                                'msg': f"Transformador '{nome}' no pipeline ANTES do split",
                                                'sugestao': 'Pipeline deve ser executado APÓS o split'
                                            })

                                    validar_params(nome, params)

                                    ctx['pipeline'].append({
                                        'tipo': 'transformador',
                                        'nome': nome,
                                        'params': params
                                    })

                                elif passo.data == 'modelo':
                                    nome = extrair_nome(passo.children[0])
                                    params = {}

                                    if len(passo.children) > 1:
                                        params = extrair_params(passo.children[1])

                                    if not ctx['split_feito']:
                                        ctx['erros'].append({
                                            'tipo': 'VAZAMENTO_DADOS',
                                            'msg': f"Pipeline com modelo '{nome}' sem split definido",
                                            'sugestao': 'Defina "split" antes do pipeline'
                                        })

                                    validar_params(nome, params)

                                    if ctx['problema']:
                                        if nome in mapa_modelos:
                                            if ctx['problema'] not in mapa_modelos[nome]:
                                                ctx['erros'].append({
                                                    'tipo': 'MODELO_INCOMPATIVEL',
                                                    'msg': f"'{nome}' não suporta {ctx['problema']}",
                                                    'sugestao': f"Use um modelo compatível com {ctx['problema']}"
                                                })

                                    ctx['pipeline'].append({
                                        'tipo': 'modelo',
                                        'nome': nome,
                                        'params': params
                                    })

                if not ctx['pipeline']:
                    ctx['erros'].append({
                        'tipo': 'PIPELINE_VAZIO',
                        'msg': 'Pipeline não pode ser vazio',
                        'sugestao': 'Adicione pelo menos um modelo ao pipeline'
                    })

            case 'avaliacao':
                metrica = extrair_nome(no.children[0])

                if not ctx['problema']:
                    ctx['erros'].append({
                        'tipo': 'DEPENDENCIA_AUSENTE',
                        'msg': 'Problema deve ser definido antes da métrica',
                        'sugestao': 'Defina "problem" antes de "evaluate"'
                    })
                else:
                    validas = metricas_validas.get(ctx['problema'], set())
                    if metrica not in validas:
                        ctx['erros'].append({
                            'tipo': 'METRICA_INVALIDA',
                            'msg': f"Métrica '{metrica}' inválida para {ctx['problema']}",
                            'sugestao': f"Métricas válidas: {', '.join(validas)}"
                        })

                if not ctx['pipeline']:
                    ctx['erros'].append({
                        'tipo': 'DEPENDENCIA_AUSENTE',
                        'msg': 'Nenhum modelo definido para avaliação',
                        'sugestao': 'Defina um modelo antes de "evaluate"'
                    })

                ctx['metrica'] = metrica

            case _:
                if hasattr(no, 'children'):
                    for filho in no.children:
                        percorrer(filho)

    percorrer(tree)

    if not ctx['pipeline']:
        ctx['erros'].append({
            'tipo': 'SEM_MODELO',
            'msg': 'Nenhum modelo definido',
            'sugestao': 'Defina um modelo usando pipeline ou diretamente'
        })

    if not ctx['metrica']:
        ctx['warnings'].append({
            'tipo': 'SEM_AVALIACAO',
            'msg': 'Nenhuma métrica de avaliação definida',
            'sugestao': 'Use "evaluate" para avaliar o modelo'
        })

    if ctx['pipeline'] and not ctx['split_feito']:
        ctx['erros'].append({
            'tipo': 'SEM_SPLIT',
            'msg': 'Nenhum split definido para treino/teste',
            'sugestao': 'Defina "split" antes do modelo'
        })

    if ctx['pipeline']:
        ultimo = ctx['pipeline'][-1]
        if ultimo['tipo'] != 'modelo':
            ctx['erros'].append({
                'tipo': 'PIPELINE_INCOMPLETO',
                'msg': f"Pipeline termina com transformador '{ultimo['nome']}', mas deve terminar com um modelo",
                'sugestao': 'Adicione um modelo após os transformadores'
            })

    if ctx['scaler'] and ctx['pipeline']:
        tem_standard_scaler = any(
            passo['tipo'] == 'transformador' and passo['nome'] == 'StandardScaler'
            for passo in ctx['pipeline']
        )
        if ctx['scaler'] == 'standard' and tem_standard_scaler:
            ctx['warnings'].append({
                'tipo': 'REDUNDANCIA',
                'msg': "Scaler 'standard' e transformador 'StandardScaler' são redundantes",
                'sugestao': 'Remova um deles para evitar duplicação'
            })

    return ctx