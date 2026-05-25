# Лабораторная работа 2. Обучение языковой модели

GPT-like языковая модель обучается на packed dataset и собственном
BPE-токенизаторе, подготовленных для `wikitext`.

## Архитектура

Модель реализована в `src/models/gpt.py`:

- синусоидальное позиционное кодирование с обнулением позиции в начале каждого packed-сегмента;
- многоголовое masked self-attention с block mask между независимыми объектами;
- FFN-блок `Linear -> GELU -> Linear`;
- post-norm transformer layer: `LayerNorm(x + Attention(x))`, затем `LayerNorm(z + FFN(z))`;
- LM-head, возвращающий логиты без дополнительного `Softmax`;
- loss mask, исключающая переходы между разными packed-сегментами и `<PAD>`.

Основная конфигурация находится в `configs/lab2_gpt.yaml`:

```yaml
vocab_size: 1000
max_seq_len: 512
d_model: 256
n_layers: 6
n_heads: 4
d_ff: 1024
dropout: 0.05
batch_size: 8
max_epochs: 20
learning_rate: 0.0005
warmup_steps: 300
gradient_clip_val: 1.0
```

Словарь `1000` выбран для обучения небольшой модели с нуля на `wikitext`.
В первом эксперименте словарь `8000` дал `val_perplexity=208.607`.

## Инфраструктура

- `src/training/data_module.py` - чтение packed dataset и train/validation split;
- `src/training/lightning_module.py` - обучение, perplexity, warm-up, scheduler и gradient norms;
- `src/training/generation.py` - генерация из checkpoint;
- `cli/lab2.py` - команды обучения и инференса;
- TensorBoard logging и опциональная интеграция ClearML;
- `ModelCheckpoint` с сохранением лучшей модели по `val_perplexity`.

## Запуск

```bash
pip install -r requirements.txt
python -m cli.lab2 train --config configs/lab2_gpt.yaml
```

Продолжить обучение:

```bash
python -m cli.lab2 train \
  --config configs/lab2_gpt.yaml \
  --resume-from-checkpoint checkpoints/lab2_vocab1000/last.ckpt
```

Посмотреть TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Сгенерировать текст из лучшего checkpoint:

```bash
python -m cli.lab2 generate \
  --config configs/lab2_gpt.yaml \
  --checkpoint checkpoints/lab2_vocab1000/final-epoch=19-val_perplexity=8.30.ckpt \
  --prompt "The history of artificial intelligence"
```

## Результаты

После исправления инициализации модели и перехода на BPE-словарь размера
`1000` получены результаты:

```text
train_loss: 2.043
train_perplexity: 7.712
val_loss: 2.117
val_perplexity: 8.303
```

Порог задания `val_perplexity <= 30` достигнут.

Лучший checkpoint:

```text
checkpoints/lab2_vocab1000/final-epoch=19-val_perplexity=8.30.ckpt
```

Пример генерации:

```text
The history of artificial intelligence is a partial location in the city , which is now known for the city 's fossils , can be taken from the Capitol Columbia .
```
