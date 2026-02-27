# Ozon Internship Prep Plan (DS Recommendations)

## Цель

Собрать публичный проект, который демонстрирует навыки под стек вакансии:
`Python + PyTorch + GPU retrieval + Triton + Airflow + базовый SQL`.

## Артефакты, которые стоит показать на собеседовании

1. **Качество retrieval**
- Таблица: `Recall@1/10/100`, `MRR`.
- Описание постановки задачи и сплитов (как исключали утечки).

2. **Скорость и нагрузка**
- Таблица: `p50/p95 latency`, `throughput` для:
  - exact search
  - ANN search
  - до/после оптимизаций (batching/fp16)

3. **GPU инференс**
- Отчёт о bottleneck: токенизация, модель, поиск.
- Базовые рекомендации, почему выбран такой trade-off.

4. **Деплой**
- FastAPI сервис для `/embed` и `/search`.
- Отдельный прототип Triton (если успеваем).

5. **Оркестрация и data-процессы**
- Airflow DAG: tokenization -> train -> embed_corpus -> build_index -> publish.
- SQL-отчёт по метрикам прогона (пусть даже на локальной SQLite/Postgres).

## План на 2-3 недели

### Неделя 1
- End-to-end baseline: MP3 -> tokens -> embeddings -> top-K.
- Первые метрики quality + latency.

### Неделя 2
- Улучшения модели и индекса.
- Сравнение exact vs ANN.
- FastAPI endpoint для реальной интеграции в плеер.

### Неделя 3 (опционально)
- Triton serving.
- Airflow DAG + SQL отчёт.
- Привести README к виду "мини-исследование + мини-прод".

## Что учить параллельно (точечно)

1. PyTorch:
- `Dataset/DataLoader`, mask/padding, contrastive loss.

2. SQL:
- `GROUP BY`, `window functions`, агрегации для offline-метрик.

3. Системная часть:
- p50/p95, throughput, batch size trade-offs, memory/latency компромисс.

