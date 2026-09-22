# Automated Image Analysis for *Daphnia similis*
 
Ensaios de nanotoxicidade com *Daphnia similis* produzem grandes volumes de imagens de placas de Petri, a partir das quais é necessário obter informações morfológicas dos organismos, como seu tamanho. A medição manual demanda tempo, está sujeita à variabilidade entre operadores e dificulta a análise padronizada de grandes conjuntos de dados. Este projeto, desenvolvido por estudantes da Ilum Escola de Ciência (CNPEM), propõe uma biblioteca em Python para automatizar essa análise.
 
A biblioteca recebe as imagens das placas e retorna as medidas de cada indivíduo, seguindo as etapas abaixo:
 
1. **Detecção** dos indivíduos presentes na imagem por um modelo de detecção de objetos;
2. **Recorte** da região de cada detecção;
3. **Processamento de imagem**, com filtros e segmentação para isolar a *Daphnia* do fundo;
4. **Medição** por ajuste de elipse, da qual são extraídos os eixos maior e menor;
5. **Exportação** dos resultados em CSV, com a identificação da imagem de origem de cada indivíduo.
> **Projeto em desenvolvimento.** Atualmente, o repositório contém o treino e a avaliação dos modelos de detecção (etapa 1). As >demais etapas serão incorporadas ao longo do projeto.
 
## Organização do repositório
 
Para a etapa de detecção, foram treinadas e comparadas duas arquiteturas: o **YOLO11n**, um modelo leve e de inferência rápida, e o **Faster R-CNN**, um detector de dois estágios. Cada uma possui sua própria pasta em `training/`:
 
```
training/
├── yolo/
└── faster_rcnn/
```
 
As duas pastas seguem a mesma organização. O `common.py` realiza a leitura das anotações exportadas do CVAT e a divisão das imagens em conjuntos de treino, validação e teste. Esse arquivo é idêntico nas duas pastas, de modo que ambos os modelos sejam treinados e avaliados sobre os mesmos dados. Os scripts `train_*.py` e `evaluate_*.py` executam o treino e a avaliação; os scripts `submit_*.sh` submetem essas execuções ao cluster via SLURM; e o `environment.yml` descreve o ambiente conda correspondente.
 
## Modelos treinados
 
Os pesos dos modelos estão disponíveis na página de [Releases](https://github.com/Luiza160/-Automated-Image-Analysis-for-Daphnia-similis/releases).
 
| Modelo | Arquivo | mAP50 | mAP50-95 | F1 |
|---|---|---|---|---|
| YOLO11n | `yolo11n_best.pt` | 0,958 | 0,622 | 0,920 |
| Faster R-CNN (ResNet50-FPN v2) | `fasterrcnn_best.pth` | 0,959 | 0,589 | 0,962 |
 
As métricas foram calculadas em um conjunto de teste mantido fora de todo o processo de treino, inclusive do critério de parada. Os dois modelos apresentaram desempenho semelhante. Como o conjunto de teste contém apenas 8 imagens, porém, uma ou duas detecções a mais ou a menos já alteram sensivelmente o F1, e a comparação entre os modelos deve ser interpretada com cautela.
 
## Reprodução dos experimentos
 
O dataset não está incluído no repositório. Para reproduzir o treino, são necessárias as imagens anotadas no CVAT e exportadas em formato COCO.

O ambiente de cada modelo pode ser criado, a partir da pasta correspondente, com:
 
```bash
conda env create -f environment.yml
```
 
Em seguida, devem ser ajustados, no início dos scripts `submit_*.sh`, o caminho do arquivo exportado do CVAT (`CVAT_ZIP`) e o diretório onde os resultados serão salvos (`WORKDIR`). O treino e a avaliação são então submetidos ao cluster:
 
```bash
sbatch submit_train_yolo.sh
sbatch submit_eval_yolo.sh
```
 
Ao final do treino, é gerado um arquivo `run_config.json` com as configurações utilizadas. A avaliação lê esse arquivo para reproduzir exatamente a mesma divisão das imagens; por isso, o `WORKDIR` deve ser o mesmo nos dois scripts.
 
### Observação sobre o checkpoint do YOLO
 
Para o YOLO, recomenda-se utilizar o arquivo `best_real.pt` em vez do `best.pt`. O Ultralytics salva no `best.pt` uma média móvel exponencial dos pesos (EMA), que funciona bem em datasets grandes, mas não converge adequadamente com poucas imagens, como no caso deste projeto. Por esse motivo, o `train_yolo.py` salva separadamente, no `best_real.pt`, os pesos da melhor época.
 
