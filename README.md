# CL_ml: Audio Retrieval для CloudTune

Репозиторий для ML-части плеера CloudTune: от токенов аудио до retrieval-сервиса с поиском похожих треков.

## Цель проекта

Собрать практичный retrieval pipeline (как mini production):

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
- Локально: подготовка кода, запуск сервиса, лёгкие тесты, инференс/индексация

## Целевая архитектура

### Offline (подготовка данных/индекса)

- MP3 -> MOSS токены
- Обучение embedder модели на токенах
- Прогон корпуса в embedding-вектора
- Сборка индекса FAISS
- Публикация артефактов (модель + индекс)

### Online (интеграция с плеером)

- Вход: `track_id` или аудио
- Расчёт embedding
- Поиск соседей в ANN-индексе
- Возврат top-K похожих треков (+ фильтры)

## Плановая структура репозитория

```text
audio-retrieval/
  README.md
  pyproject.toml / requirements.txt
  configs/
    train.yaml
    index.yaml
    triton_model_config.pbtxt
  data/
    raw_audio/
    tokens/
    splits/
  src/
    tokenizer/
      moss_tokenize.py
    dataset/
      token_dataset.py
      augmentations.py
    model/
      embedder.py
      losses.py
    train/
      train_contrastive.py
      eval_retrieval.py
    index/
      build_faiss.py
      search_faiss.py
      eval_latency.py
    service/
      api.py
      schemas.py
      settings.py
  triton/
    model_repository/
  airflow/
    dags/
      pipeline.py
  scripts/
```

## Роадмап (этапы)

### Этап 0. Подготовка окружения (0.5-1 день)

- Python 3.10-3.12
- PyTorch + базовые зависимости (`numpy`, `torchaudio`, `faiss`, `fastapi`, `uvicorn`)
- Установка зависимостей MOSS-токенизатора
- Готовность: `python -m src.tokenizer.moss_tokenize --help`

### Этап 1. Данные и постановка retrieval-задачи (1 день)

- Определить позитивы: один трек + его аугментации
- Негативы: другие треки в батче
- Сделать `train/val/test` сплиты

### Этап 2. Оффлайн токенизация MOSS (1-2 дня)

- Скрипт MP3 -> токены, сохранение в `data/tokens/`
- Токенизация выполняется один раз, дальше переиспользуется
- Логирование времени токенизации

### Этап 3. Датасет и загрузка батчей (0.5-1 день)

- `Dataset` возвращает `tokens_a`, `tokens_b`, `track_id`
- Padding/mask для переменной длины
- `DataLoader` с рабочими батчами

### Этап 4. Модель `tokens -> embedding` (1-2 дня)

- Token embedding -> Transformer Encoder -> pooling -> projection
- Размер embedding: 256/512
- L2-нормализация для cosine similarity

### Этап 5. Contrastive обучение (2-4 дня)

- InfoNCE / NT-Xent
- Логи: train loss, val Recall@K, speed
- Сохранение checkpoint и кривых обучения

### Этап 6. Retrieval оценка (1 день)

- Query: аугментированный кусок трека
- Target: попадание того же `track_id` в top-K
- Метрики: Recall@1/10/100, опционально MRR

### Этап 7. Индексация и trade-off (1-2 дня)

- Baseline: `IndexFlatIP` (точный поиск)
- ANN: IVF (+ PQ опционально)
- Сравнение: latency vs recall

### Этап 8. API сервис (1-2 дня)

- `POST /embed`
- `GET /search?track_id=...&k=...`
- `POST /search_by_audio`
- `GET /health`

### Этап 9. Оптимизация инференса (1-2 дня)

- Batch inference
- FP16 (при CUDA)
- Профилирование bottleneck
- Бенчмарки p50/p95 + throughput

### Этап 10. Triton (опционально, 2-4 дня)

- Экспорт модели (ONNX/TorchScript)
- Dynamic batching, instance groups
- Сравнение производительности с FastAPI

### Этап 11. Airflow пайплайн (опционально, 1-3 дня)

- DAG: tokenize -> train -> export -> embed_corpus -> build_index -> publish

## Ближайшие цели (следующие 7-10 дней)

1. Поднять каркас репозитория (`src/`, `configs/`, `scripts/`, `data/`).
2. Добавить рабочий `moss_tokenize.py` и прогнать токенизацию на маленьком наборе треков.
3. Зафиксировать формат токенов и сплиты (`train/val/test`).
4. Реализовать `TokenDataset` + аугментации + первый `DataLoader`.
5. Сделать baseline embedder и dry-run обучения на Colab (1-2 эпохи).
6. Посчитать первые offline-метрики retrieval (Recall@10/100) на маленьком тесте.

## Критерии готовности MVP

- Есть end-to-end pipeline: MP3 -> токены -> embedding -> top-K поиск
- Есть базовые метрики качества и задержки
- Есть API endpoint для “похожих треков”
- Есть план следующей оптимизации (ANN/Triton/батчинг)
