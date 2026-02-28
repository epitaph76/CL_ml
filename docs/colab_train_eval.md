# Colab: Training + Retrieval Evaluation

Ниже минимальный сценарий для Google Colab (`Python 3.10`, GPU Runtime).

## 1) Клонировать репозиторий и поставить зависимости

```bash
!git clone https://github.com/epitaph76/CL_ml.git
%cd /content/CL_ml
!pip install -r requirements.txt
```

Проверка CLI:

```bash
!python -m src.train.train_contrastive --help
!python -m src.index.evaluate_retrieval --help
```

## 2) Подготовить данные

Вариант A: если токены уже готовы локально/на Drive, просто скопировать их в `data/tokens`.

Вариант B: токенизировать аудио прямо в Colab:

```bash
!python -m src.tokenizer.moss_tokenize \
  --input-root data/raw_audio \
  --output-root data/tokens \
  --device auto
```

Построить сплиты:

```bash
!python -m src.dataset.build_splits \
  --tokens-root data/tokens \
  --output-root data/splits \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

## 3) Обучить baseline embedder

Полный запуск:

```bash
!python -m src.train.train_contrastive \
  --config configs/train.yaml \
  --device auto \
  --output-dir data/checkpoints
```

Короткий smoke-запуск:

```bash
!python -m src.train.train_contrastive \
  --config configs/train.yaml \
  --device auto \
  --max-steps-per-epoch 5 \
  --output-dir data/checkpoints_smoke
```

## 4) Посчитать retrieval-метрики

Exact search:

```bash
!python -m src.index.evaluate_retrieval \
  --config configs/train.yaml \
  --checkpoint data/checkpoints/best.pt \
  --split val \
  --topk 1,10,100 \
  --device auto \
  --output-json data/reports/val_exact.json
```

Exact + FAISS:

```bash
!python -m src.index.evaluate_retrieval \
  --config configs/train.yaml \
  --checkpoint data/checkpoints/best.pt \
  --split val \
  --topk 1,10,100 \
  --device auto \
  --use-faiss \
  --output-json data/reports/val_exact_faiss.json
```

Ожидаемая таблица в stdout:
- `recall@1`
- `recall@10`
- `recall@100`
- `mrr`

## 5) Сохранить артефакты на Google Drive (опционально)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/CL_ml_runs/run1
!cp -r data/checkpoints /content/drive/MyDrive/CL_ml_runs/run1/
!cp -r data/reports /content/drive/MyDrive/CL_ml_runs/run1/
!cp -r configs /content/drive/MyDrive/CL_ml_runs/run1/
```

