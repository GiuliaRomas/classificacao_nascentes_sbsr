# Classificação de nascentes

Este repositório reúne os notebooks e módulos Python usados para classificar imagens de nascentes em quatro níveis de preservação.

## Classes

- Preservada
- Moderada
- Baixa
- Não Preservada

## Metodologia

O modelo implementado em `models/convnext_swin.py` combina os extratores de características **ConvNeXt-Tiny** e **Swin-Tiny**, com pesos pré-treinados no ImageNet. Um mecanismo de *attention pooling* resume as características espaciais de cada backbone, em seguida, elas são concatenadas e encaminhadas ao classificador.

As configurações padrão em `configs/config.py` definem imagens de 128 × 128 pixels, lotes de 32 imagens, semente aleatória 42 e até 50 épocas. Os backbones são pré-treinados e congelados por padrão. O treinamento usa pesos por classe e *early stopping*. O desempenho é medido por acurácia, precisão, recall e F1-macro.

A validação cruzada usa 5 folds estratificados e agrupados por UGRHI (`StratifiedGroupKFold`), mantendo as unidades espaciais separadas entre treino e validação.

## Estrutura do projeto

```text
codigos_modelos/
├── checkpoints/       # Pesos dos modelos treinados
├── configs/           # Configurações de dados e treinamento
├── data/
│   ├── amostras/      # Shapefile com os grupos espaciais
│   └── dataset/       # Imagens organizadas por classe
├── logs/              # Registros de treinamento
├── models/            # Arquiteturas dos modelos
├── notebooks/         # Notebooks de treinamento e otimização
├── results/           # Métricas, gráficos e erros de classificação
└── utils/             # Dataset, treinamento, avaliação e métricas
```

## Dados

```text
data/dataset/
├── treino/
│   ├── Preservada/
│   ├── Moderada/
│   ├── Baixa/
│   └── Não Preservada/
└── teste/
    ├── Preservada/
    ├── Moderada/
    ├── Baixa/
    └── Não Preservada/
```

Os nomes dos diretórios devem corresponder às classes configuradas. Para a validação espacial, `data/amostras/amostra_final.shp` deve conter os campos `fid`, `ugrhi_id` e `ugrhi_nome`, com um FID correspondente ao identificador de cada imagem.

## Resultado

O resultado final obtido foi:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,750 |
| Precisão macro | 0,737 |
| Revocação macro | 0,740 |
| F1-macro | 0,738 |

O teste contém 388 imagens.