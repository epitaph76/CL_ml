# CL_ml: Audio Retrieval для CloudTune

Репозиторий для ML-части плеера CloudTune: от токенизации аудио до retrieval-поиска похожих треков.

## Статус на сейчас

Сделан старт по Этапам 0-2:

- Этап 0: подготовлен каркас проекта, зависимости и конфиги.
- Этап 1: добавлен скрипт формирования `train/val/test` сплитов.
- Этап 2: добавлен рабочий CLI для токенизации аудио через MOSS.

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
  src/
    tokenizer/
      moss_tokenize.py
    dataset/
      build_splits.py
    model/
    train/
    index/
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

- сканирует аудио в `--input-root`
- грузит MOSS tokenizer из HuggingFace
- приводит аудио к sample rate модели
- токенизирует трек (или чанки)
- сохраняет токены в `data/tokens` (`.pt` или `.npz`)

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

## Ближайшие цели (следующие шаги)

1. Добавить `TokenDataset` + аугментации (`src/dataset/token_dataset.py`, `src/dataset/augmentations.py`).
2. Реализовать baseline-модель `tokens -> embedding` (`src/model/embedder.py`).
3. Запустить первое контрастивное обучение в Colab и сохранить checkpoint.
4. Сделать базовую offline оценку retrieval (`Recall@10/100`).
