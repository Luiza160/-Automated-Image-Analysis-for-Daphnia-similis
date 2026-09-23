# Automated Image Analysis for *Daphnia similis*

Ensaios de nanotoxicidade com *Daphnia similis* produzem grandes volumes de imagens de placas de Petri, a partir das quais é necessário obter informações morfológicas dos organismos, como seu tamanho. A medição manual demanda tempo, está sujeita à variabilidade entre operadores e dificulta a análise padronizada de grandes conjuntos de dados. Este projeto, desenvolvido por estudantes da Ilum Escola de Ciência (CNPEM), propõe uma biblioteca em Python para automatizar essa análise.

A biblioteca recebe as imagens das placas e retorna as medidas de cada indivíduo, seguindo as etapas abaixo:

1. **Detecção** dos indivíduos presentes na imagem por um modelo de detecção de objetos;
2. **Recorte** da região de cada detecção, com uma margem proporcional ao tamanho da caixa;
3. **Processamento de imagem**, com filtros aplicados apenas aos recortes que não são medidos corretamente sem eles;
4. **Medição** com o [Daphnia Ruler](https://github.com/nelstevens/The-Daphnia-ruler), que segmenta o organismo e ajusta uma elipse ao corpo para extrair seu comprimento;
5. **Exportação** dos resultados em CSV, com a identificação da imagem de origem de cada indivíduo.

> **Projeto em desenvolvimento.** Atualmente, o repositório contém o treino e a avaliação dos modelos de detecção (etapa 1) e uma primeira versão do pipeline de medição, que combina detecção, recorte e Daphnia Ruler, ainda sem filtros de processamento. As demais etapas serão incorporadas ao longo do projeto.

## Organização do repositório

Para a etapa de detecção, foram treinadas e comparadas duas arquiteturas: o **YOLO11n**, um modelo leve e de inferência rápida, e o **Faster R-CNN**, um detector de dois estágios. Cada uma possui sua própria pasta em `training/`.

A pasta `pipeline/` contém a avaliação do pipeline completo sem filtros. Ela identifica quais *Daphnias* já são medidas corretamente pelo Daphnia Ruler, para que os filtros de processamento sejam aplicados apenas às restantes.

```
training/
├── yolo/
└── faster_rcnn/
pipeline/
├── avaliar_pipeline_completo.py
├── common.py
└── slurm/
```

## Referências

- Daphnia Ruler: https://github.com/nelstevens/The-Daphnia-ruler
