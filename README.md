# modern-ai-architecture

Код для лабораторной работы 1 "Подготовка данных".

## Что реализовано

- Загрузка одного WARC-файла Common Crawl.
- Конвертация HTML-страниц из WARC в текстовый JSONL.
- Очистка текста: удаление HTML, нормализация Unicode и пробелов, фильтрация языков, пустых и слишком коротких объектов, разбиение длинных объектов.
- Оценка энтропии каждого объекта через GPT-2 из `transformers`.
- Оценка информационной плотности датасета как взвешенного среднего по числу токенов.
- Удаление полных дубликатов и объектов с экстремально низкой/высокой энтропией.
- Символьная, словарная и BPE-токенизация.
- Загрузка и обработка `wikitext`.
- Packed batching с маской сегментов: `0` для `<PAD>`, `1..N` для склеенных объектов.

## Структура

- `cli/lab1.py` - CLI для запуска всех этапов.
- `src/data/` - загрузка, WARC-парсинг, очистка, wikitext, packed batching.
- `src/models/entropy.py` - оценка энтропии GPT-2.
- `src/tokenization/` - char/word/BPE токенизаторы.
- `configs/default.json` - основные параметры.

## Команды запуска

Создать окружение и установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запустить пайплайн для Common Crawl:

```bash
python -m cli.lab1 download-cc
python -m cli.lab1 warc-to-text --limit 1000
python -m cli.lab1 clean
python -m cli.lab1 score-entropy
python -m cli.lab1 filter-quality
python -m cli.lab1 tokenize
```

По умолчанию команда `download-cc` сама берет реальный WARC-путь из манифеста
`warc.paths.gz` для снапшота `CC-MAIN-2024-42`. Можно выбрать другой снапшот
или другой файл:

```bash
python -m cli.lab1 download-cc --snapshot CC-MAIN-2024-10 --warc-index 0
```

Запустить те же этапы для `wikitext`, используя BPE-токенизатор, обученный на Common Crawl:

```bash
python -m cli.lab1 prepare-wikitext
python -m cli.lab1 score-entropy --input data/processed/wikitext_clean.jsonl --output data/processed/wikitext_entropy.jsonl
python -m cli.lab1 filter-quality --input data/processed/wikitext_entropy.jsonl --output data/processed/wikitext_quality.jsonl
python -m cli.lab1 tokenize --input data/processed/wikitext_quality.jsonl --bpe-load data/processed/common_crawl_bpe.json
python -m cli.lab1 pack --input data/processed/wikitext_quality.jsonl --tokenizer data/processed/common_crawl_bpe.json
```

Для быстрой проверки можно уменьшить объем данных:

```bash
python -m cli.lab1 warc-to-text --limit 100
python -m cli.lab1 prepare-wikitext --limit 500
```
