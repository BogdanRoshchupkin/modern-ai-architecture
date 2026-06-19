# Лабораторная работа 3. Flash Attention

Реализация маскированного FlashAttention находится в
`src/backend/flash_attention.py` и `src/backend/flash_attention_triton.py`.
Код работает с тензорами формата
`[batch, heads, seq_len, head_dim]` и `segment_ids` формата
`[batch, seq_len]`. Маска одновременно учитывает:

- causal attention: токен не смотрит в будущие позиции;
- packed-сегменты: токены из разных объектов не attend друг к другу;
- padding: сегмент `0` полностью исключается из attention.

### Что реализовано

- `torch_masked_attention` - reference-реализация на PyTorch для сравнения;
- `masked_flash_attention_forward` - блочный forward FlashAttention;
- fully masked блоки `S_i,j` пропускаются через `continue`;
- online softmax использует `row_max`, `row_sum` и аккумулятор `acc`, поэтому
  полная score-матрица `[seq_len, seq_len]` не хранится;
- `masked_flash_attention_backward` - явная формула backward для `q`, `k`, `v`;
- `MaskedFlashAttentionFunction` - собственный `torch.autograd.Function`;
- `MaskedFlashAttention` - `torch.nn.Module`-обертка;
- `triton_masked_flash_attention` - CUDA/Triton backend с `@triton.jit`
  forward и backward kernels;
- `TritonMaskedFlashAttentionFunction` - `torch.autograd.Function` для
  Triton kernels;
- `TritonMaskedFlashAttention` - `torch.nn.Module`-обертка для CUDA/Triton;
- `benchmark_flash_attention` и `cli/lab3.py` - сравнение времени и памяти
  с torch-реализацией.

### Тесты

Тесты находятся в `tests/test_flash_attention.py`.

Проверяется:

- совпадение forward с torch reference через `torch.testing.assert_close`;
- совпадение backward-градиентов по `q`, `k`, `v`;
- отдельный backward wrapper;
- работа `torch.nn.Module`-обертки.
- CUDA/Triton forward и backward tests, которые автоматически пропускаются
  на машинах без CUDA/Triton.

Проверить корректность:

```bash
python -m pytest tests/test_flash_attention.py -q
```

Запустить полный набор тестов:

```bash
python -m pytest -q
```

Ожидаемый результат:

```text
CPU-only: 4 passed, 2 skipped for FlashAttention tests
```

### Бенчмарк

Запустить бенчмарк:

```bash
python -m cli.lab3 --seq-len 512 --head-dim 64 --heads 4 --batch-size 2 --repeats 10
```

Запустить Triton backend в Colab T4:

```bash
python -m cli.lab3 \
  --backend triton \
  --device cuda \
  --seq-len 512 \
  --head-dim 64 \
  --heads 4 \
  --batch-size 2 \
  --repeats 10
```

Пример результата на CPU:

```text
torch attention median: 5.536 ms
flash attention median: 9.630 ms
speedup: 0.57x
torch score matrix memory: 8.00 MB
flash score block memory: 0.12 MB
score-memory reduction: 64.00x
```

Интерпретация:

- `speedup < 1` на CPU означает, что блочная Python/PyTorch-реализация
  медленнее оптимизированной torch-операции;
- ключевой результат для FlashAttention здесь - уменьшение памяти под
  score-матрицу: вместо полной матрицы используется только текущий блок;
- в примере память уменьшилась с `8.00 MB` до `0.12 MB`, то есть в `64x`;
- для реального ускорения по времени нужен CUDA/Triton GPU backend.

Для удобного запуска на Google Colab T4 есть notebook:

```text
notebooks/lab3_colab_cuda_benchmark.ipynb
```

Он проверяет CUDA, ставит зависимости, запускает CUDA/Triton тесты и строит
таблицу/графики benchmark для `torch-blocked` и `triton` backend.

# Лабораторная работа 4. Инференс

В ЛР4 модель из предыдущих работ расширена для инференса:

- реализован GQA через отдельные `q_proj`, `k_proj`, `v_proj`;
- добавлен параметр `n_kv_heads` в YAML-конфиг;
- реализован KV-cache для autoregressive inference;
- добавлен CLI для обучения и генерации;
- подготовлен Colab notebook для полного запуска.

## GQA

Конфиг находится в `configs/lab4_gqa.yaml`.

Ключевые параметры:

```yaml
n_heads: 4
n_kv_heads: 2
```

Это означает, что query-heads остаются в количестве `4`, а key/value-heads
становятся общими для групп query-heads. За счет этого уменьшается размер
KV-проекций и KV-cache.

## KV-cache

Во время генерации prompt сначала проходит через модель целиком и создает
cache по каждому transformer block. Затем каждый новый токен обрабатывается
как один шаг:

```text
past K/V + new token -> next logits + updated K/V
```

Это ускоряет autoregressive inference, потому что модель не пересчитывает
ключи и значения для всего префикса заново.

## Команды

Проверить GQA и KV-cache:

```bash
python -m pytest tests/test_gqa_kv_cache.py -q
```

Быстро проверить train loop:

```bash
python -m cli.lab4 train --config configs/lab4_gqa.yaml --fast-dev-run
```

Запустить обучение:

```bash
python -m cli.lab4 train --config configs/lab4_gqa.yaml
```

Сгенерировать текст с KV-cache:

```bash
python -m cli.lab4 generate \
  --config configs/lab4_gqa.yaml \
  --checkpoint checkpoints/lab4_gqa/final-epoch=XX-val_perplexity=YY.ckpt \
  --prompt "The history of artificial intelligence"
```

Сгенерировать текст без KV-cache для сравнения:

```bash
python -m cli.lab4 generate \
  --config configs/lab4_gqa.yaml \
  --checkpoint checkpoints/lab4_gqa/final-epoch=XX-val_perplexity=YY.ckpt \
  --prompt "The history of artificial intelligence" \
  --no-kv-cache
```

## Colab

Для запуска на T4 подготовлен notebook:

```text
notebooks/lab4_colab_inference.ipynb
```

Он по ячейкам:

- клонирует ветку `lab4`;
- ставит зависимости;
- создает `.env` с `ROOT_DIR`;
- готовит Wikitext, BPE tokenizer и packed dataset;
- запускает GQA/KV-cache tests;
- обучает модель;
- показывает TensorBoard;
- генерирует текст из лучшего checkpoint;
- копирует checkpoints/logs/config в Google Drive.

Критерии PDF:

- `val_perplexity <= 40` - частичный балл за обучение;
- `val_perplexity <= 25` - полный балл за обучение.
