# Análise automatizada de imagens de *Daphnia similis* e peixe-zebra

Projeto de Letícia Nunes de Souza Andrade e Luiza Davoli, da Ilum Escola de Ciência (CNPEM). O código automatiza a detecção e a medição de organismos em imagens de ensaios de nanotoxicidade.

| Caminho | Conteúdo |
| --- | --- |
| `training/yolo/` | Preparação do conjunto CVAT, treino e avaliação do YOLO11n |
| `training/faster_rcnn/` | Treino e avaliação do Faster R-CNN com ResNet-50 FPN v2 |
| `pipeline/` | Avaliação da cascata em conjuntos anotados, com Daphnia Ruler e Real-ESRGAN |
| `app/` | Cascata para lotes de imagens, filtros e versão original da interface Streamlit |
| `direct/` | Medição direta da Daphnia, medição de peixe-zebra, validação com ImageJ e interfaces revisadas |
| `analysis/` | Gráficos a partir dos CSVs de treinamento |

## Medição direta e interfaces

Crie um ambiente Python 3.12 e instale as dependências a partir da raiz do repositório.

```bash
python -m pip install -r direct/requirements.txt
```

Use a configuração de captura para calibrar a medida. O comprimento é registrado em pixels e, quando `--px-por-mm` é informado, também em milímetros.

### Daphnia

O detector localiza cada indivíduo. O recorte é padronizado pela diagonal da caixa de detecção. O código estima um plano de fundo por canal de cor, aplica o limiar de Otsu e refina a máscara com operações morfológicas. O comprimento é calculado ao longo do eixo longitudinal, com exclusão das pontas estreitas. O CSV indica as medidas que exigem conferência e o motivo da sinalização.

```bash
python direct/main.py --entrada data/placas.zip --model-type yolo --checkpoint weights/yolo11n_best.pt --workdir runs/daphnia-yolo --salvar-etapas
python direct/main.py --entrada data/placas.zip --model-type fasterrcnn --checkpoint weights/fasterrcnn_best.pth --workdir runs/daphnia-faster
```

A entrada pode ser uma foto, uma pasta ou um ZIP. Cada execução grava `medidas_daphnia.csv` e as imagens anotadas na pasta de trabalho. Use uma pasta de trabalho própria para cada lote.

### Peixe-zebra

A rotina estima o fundo, segmenta a larva por contraste e usa a maior distância entre os pontos dos contornos selecionados como comprimento em linha reta. O lote é processado com uma larva por imagem.

```bash
python direct/medir_zebrafish.py data/zebrafish
```

### Interfaces locais

Coloque os checkpoints em `weights/` com os nomes dos exemplos acima. Execute o Streamlit de dentro de `direct/` para carregar o tema do projeto.

```bash
cd direct
python -m streamlit run app_completo.py
python app.py
```

São dois comandos de lançamento independentes. A interface Flask abre em `http://127.0.0.1:5000`. O Streamlit informa o endereço ao iniciar.

## Cascata com Daphnia Ruler

Na cascata, as imagens são detectadas e recortadas. O Ruler mede primeiro os recortes sem filtro. Os casos pendentes são encaminhados às etapas seguintes na ordem informada, incluindo Real-ESRGAN quando selecionado. Guarde a ordem, os limites de plausibilidade e a calibração usados em cada comparação.

```bash
conda env create -f pipeline/environment.yaml
conda activate pipeline_env
python pipeline/avaliar_pipeline_cascata.py --cvat-zip data/dataset.zip --format coco --model-type yolo --checkpoint weights/yolo11n_best.pt --workdir runs/cascata-cvat --filtros sem_filtro realesrgan --realesrgan-weights weights/RealESRGAN_x4plus.pth --realesrgan-tile 256
python app/executar_cascata.py --dataset-zip data/placas.zip --model-type yolo --checkpoint weights/yolo11n_best.pt --workdir runs/cascata-lote --filtros sem_filtro realesrgan --realesrgan-weights weights/RealESRGAN_x4plus.pth
```

O primeiro comando Python recebe uma exportação anotada do CVAT. O segundo aceita um ZIP de imagens e oferece filtros adicionais, como mediana e bilateral. A interface original da cascata requer Streamlit no ambiente ativo e é iniciada com `python -m streamlit run app/app_completo.py`.

Os ensaios de filtros adicionais ficam em `pipeline/experimentos/avaliar_filtros.py`. A rotina inclui CLAHE, nitidez, deconvolução, top-hat, black-hat, mediana, bilateral, Real-ESRGAN e HAT. Use `--help` para consultar os parâmetros e registre a sequência avaliada com `--filtros`. A deconvolução usa scikit-image. HAT requer o pacote HAT e os pesos da arquitetura selecionada.

## Treino e avaliação

Exporte as imagens e anotações do CVAT em COCO, YOLO ou Pascal VOC. O exemplo usa COCO. Ative o ambiente correspondente antes de executar cada par de comandos.

```bash
conda env create -f training/yolo/environment.yml
conda activate yolo_env
python training/yolo/train_yolo.py --cvat-zip data/dataset.zip --format coco --workdir runs/yolo --n-augmentations 2 --seed 42
python training/yolo/evaluate_yolo.py --cvat-zip data/dataset.zip --workdir runs/yolo --checkpoint runs/yolo/yolo_run/weights/best_real.pt
```

```bash
conda env create -f training/faster_rcnn/environment.yml
conda activate fasterrcnn_env
python training/faster_rcnn/train_fasterrcnn.py --cvat-zip data/dataset.zip --format coco --workdir runs/faster --seed 42
python training/faster_rcnn/evaluate_fasterrcnn.py --cvat-zip data/dataset.zip --workdir runs/faster --checkpoint runs/faster/fasterrcnn_run/best_model.pth
```

A avaliação deve usar o mesmo conjunto de dados e a configuração de divisão registrada no treino. Guarde a versão do export, o `run_config.json` e os checkpoints de cada execução. Os scripts Slurm são submetidos a partir da raiz com o ambiente ativo. Informe `CVAT_ZIP` e `WORKDIR` com caminhos absolutos. A cascata também recebe `CHECKPOINT` e `REALESRGAN_WEIGHTS`.

## Validação e gráficos

Execute `python direct/validar.py --help` para os parâmetros da comparação com linhas medidas no ImageJ. O script `direct/teste_ruler_yolo.py` permite avaliar separadamente o Ruler com caixas YOLO sem margem. Para plotar os registros de treino, use `python analysis/plot_publication_style.py --help`.

## Dados e pesos

Os scripts recebem o export CVAT e os checkpoints como entradas locais. A execução com os checkpoints da versão revisada pode ser identificada pelos hashes SHA-256 abaixo. As imagens e os pesos experimentais devem ser obtidos com as autoras.

| Arquivo | SHA-256 |
| --- | --- |
| `yolo11n_best.pt` | `e357943e8aa20fa13f851852c757d90e6a8dd0874c5a24c460861f20976a4b89` |
| `fasterrcnn_best.pth` | `58256326a8bbe1264a5cbbcd523ca9f80f72a40fb7eb6576e1e3f591d135b239` |

[Daphnia Ruler](https://github.com/nelstevens/The-Daphnia-ruler) · [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) · [Ultralytics](https://github.com/ultralytics/ultralytics)
