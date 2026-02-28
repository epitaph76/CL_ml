# CL_ml: Audio Retrieval для CloudTune

Репозиторий для ML-части плеера CloudTune: от токенизации аудио до retrieval-поиска похожих треков.

## Статус на сейчас

Сделан старт по Этапам 0-5:

- Этап 0: подготовлен каркас проекта, зависимости и конфиги.
- Этап 1: добавлен скрипт формирования `train/val/test` сплитов.
- Этап 2: добавлен рабочий CLI для токенизации аудио через MOSS.
- Этап 3: добавлен `TokenPairDataset` с аугментациями и `collate` с padding/mask.
- Этап 4: добавлена baseline-модель `TokenEmbedder` и тренинг-CLI для контрастивного обучения.
- Этап 5: добавлена offline retrieval-оценка (`Recall@1/10/100`, `MRR`) с exact и optional FAISS.

MOSS модель: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/tree/main

## Цель проекта

Собрать практичный retrieval pipeline:

1. Аудио -> токены (MOSS-Audio-Tokenizer)
2. Токены -> embedding-вектора (своя модель)
3. ANN-индекс (FAISS/HNSW)
4. Top-K поиск кандидатов
5. Метрики качества (Recall@K, MRR)
6. Метрики производительности (p50/p95 latency, throughput)
7. Сервисная интеграция в CloudTune (FastAPI, батчинг, опционально Triton)

## Ограничения окружения

- Основная машина: ноутбук
- Обучение модели: преимущественно в Google Colab
- Локально: подготовка данных, скрипты, сервис, лёгкие проверки

## Быстрый старт (Этап 0)

Рекомендуемая версия Python: `3.10-3.12`.

Важно: на этой машине сейчас `Python 3.14.3`, а часть ML-библиотек может не иметь стабильных wheel для 3.14.
Для работы лучше использовать Colab (`Python 3.10`) или локально поставить 3.11/3.12.

### 1) Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2) Проверка CLI

```bash
python -m src.tokenizer.moss_tokenize --help
python -m src.dataset.build_splits --help
```

### 3) Colab-ноутбук (единый пайплайн)

- Рекомендуемый основной ноутбук:
  `notebooks/01_moss_tokenization_colab.ipynb`
- Он покрывает весь baseline end-to-end:
  `audio -> tokens -> splits -> train -> Recall@1/10/100, MRR`.
- Прямая ссылка для запуска:
  `https://colab.research.google.com/github/epitaph76/CL_ml/blob/main/notebooks/01_moss_tokenization_colab.ipynb`
- Дополнительный (опциональный) ноутбук только для этапа train/eval:
  `notebooks/02_train_eval_colab.ipynb`
- Если видите `ModuleNotFoundError: No module named 'src'`, значит запуск идёт не из корня репозитория.
  В ноутбуке это уже исправлено через `cwd=/content/CL_ml` в `subprocess.run(...)`.
- Для этапов обучения и retrieval-оценки: `docs/colab_train_eval.md`

## Текущая структура

```text
CL_ml/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  configs/
    train.yaml
    index.yaml
  notebooks/
    01_moss_tokenization_colab.ipynb
    02_train_eval_colab.ipynb
  src/
    tokenizer/
      moss_tokenize.py
    dataset/
      build_splits.py
      augmentations.py
      token_dataset.py
    model/
      embedder.py
    train/
      train_contrastive.py
    index/
      evaluate_retrieval.py
    service/
  scripts/
    run_tokenize.ps1
```

## Этап 1: Данные и сплиты

### Что сделано

Скрипт: `src/dataset/build_splits.py`

Он:

- читает токены из `data/tokens`
- собирает уникальные `track_id`
- создаёт `train.txt`, `val.txt`, `test.txt` в `data/splits`
- пишет `summary.json` с размерами сплитов

### Зачем это нужно

Без корректного разделения на train/val/test нельзя честно измерять retrieval-качество.

### Запуск

```bash
python -m src.dataset.build_splits --tokens-root data/tokens --output-root data/splits --val-ratio 0.1 --test-ratio 0.1
```

## Этап 2: Оффлайн токенизация MOSS

### Что сделано

Скрипт: `src/tokenizer/moss_tokenize.py`

Он:

- сканирует аудио в `--input-root` (папка или одиночный файл)
- грузит MOSS tokenizer из HuggingFace
- приводит аудио к sample rate модели
- токенизирует трек (или чанки)
- сохраняет токены в `data/tokens` (`.pt` или `.npz`)
- при проблемных файлах выводит warning и продолжает (или `--fail-fast` для остановки)

### Зачем это нужно

Токенизация — тяжёлая операция. Её делаем оффлайн один раз и дальше обучаем/индексируем уже по готовым токенам.

### Запуск

```bash
python -m src.tokenizer.moss_tokenize --input-root data/raw_audio --output-root data/tokens --device auto
```

С ограничением для smoke-теста:

```bash
python -m src.tokenizer.moss_tokenize --input-root data/raw_audio --output-root data/tokens --max-files 10
```

С чанкованием треков:

```bash
python -m src.tokenizer.moss_tokenize --input-root data/raw_audio --output-root data/tokens --chunk-seconds 8
```

PowerShell shortcut:

```powershell
./scripts/run_tokenize.ps1 -InputRoot data/raw_audio -OutputRoot data/tokens -Device auto -MaxFiles 10
```

## Этап 4: Baseline embedder и обучение

### Что сделано

Скрипты:
- `src/model/embedder.py`
- `src/train/train_contrastive.py`

`TokenEmbedder` принимает токены `[B,Q,T]` + `mask [B,T]` и возвращает L2-нормированные эмбеддинги треков.

`train_contrastive.py`:
- читает `configs/train.yaml`
- строит `TokenPairDataset` по `train/val` сплитам
- обучает модель с NT-Xent (InfoNCE)
- сохраняет `best.pt`, `last.pt`, `history.json`

### Запуск обучения

```bash
python -m src.train.train_contrastive --config configs/train.yaml --device auto --output-dir data/checkpoints --batch-size 4 --num-workers 2
```

Smoke-запуск (короткий прогон):

```bash
python -m src.train.train_contrastive --config configs/train.yaml --max-steps-per-epoch 5 --batch-size 2 --num-workers 2
```

## Этап 5: Offline retrieval оценка

### Что сделано

Скрипт: `src/index/evaluate_retrieval.py`

Он:
- собирает query/corpus представления из токенов выбранного сплита
- прогоняет модель и получает эмбеддинги
- считает `Recall@K` и `MRR`
- может дополнительно посчитать FAISS (`--use-faiss`)

### Запуск оценки

```bash
python -m src.index.evaluate_retrieval --split val --topk 1,10,100 --checkpoint data/checkpoints/best.pt --use-faiss --batch-size 16
```

С сохранением отчёта:

```bash
python -m src.index.evaluate_retrieval --split val --checkpoint data/checkpoints/best.pt --output-json data/reports/val_metrics.json --batch-size 16
```

## Ближайшие цели (следующие шаги)

1. Запустить обучение в Colab/GPU на реальных токенах и получить стабильный `best.pt`.
2. Зафиксировать первую таблицу `Recall@1/10/100`, `MRR` на `val/test`.
3. Добавить baseline индексы ANN (например, IVF/HNSW) и сравнить с exact/IndexFlat.
4. Добавить замеры `p50/p95 latency` и `throughput` для поиска.

## Связка с вакансией Ozon

Что важно в вакансии и как это отражено в проекте:

- GPU векторный движок: реализуем retrieval-pipeline с метриками latency/throughput и FAISS/ANN.
- Оптимизация инференса на GPU: отдельный этап с batch inference, FP16 и p50/p95 замерами.
- Деплой моделей в прод: FastAPI baseline и отдельный этап под Triton Inference Server.
- Работа с большими данными/пайплайнами: план на Airflow DAG для автоматизации tokenization -> training -> indexing.
- Python/PyTorch: основной код проекта уже строится вокруг этого стека.

## Учебный фокус под стажировку (приоритет)

1. Retrieval-метрики и эксперименты: `Recall@K`, `MRR`, честные валидационные сплиты.
2. GPU-inference: батчинг, профилирование bottleneck, сравнение fp32/fp16.
3. Triton: экспорт модели + динамический батчинг + бенчмарк.
4. Airflow + SQL: минимальный DAG и простые SQL-выборки для отчётности по метрикам.

Детальный план: `docs/ozon_prep_plan.md`.
