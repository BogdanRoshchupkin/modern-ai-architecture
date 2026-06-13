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
