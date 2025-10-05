## Обучение
1) Загрузка датасетов

```python ul2_data_download.py --ru 10 --en 10```

датасет кода не загружается, можно вернуть раскомментив строки в скрипте

2) Инициализация модели
```
from ul2_model import setup_modernbert_ul2

model_small, tokenizer_small, collator_small = setup_modernbert_ul2(
    encoder_model_path="jhu-clsp/mmBERT-base",
    output_dir="./mmBERT_base_ul2_e22_d22_v2",
    num_decoder_layers=22,
    share_rope_config=True,
    use_bf16=True,
)
```

3) Обучение

```bash test_lion.sh &> t5_large_train.logs```

## Мысли и дополнения

Тренировочные данные (энкодер и декодер) дополнительно оборачиваются в токены начала и конца предложения. Это не стандартная процедура, в обычных seq2seq оборачивается только выход декодера. Зачем? Возможно это даст буст по качеству, ведь у нас идет инициализация чистым энкодером который "привык" видеть текст в таком формате.

Rope - сейчас одинаковый для энкодера и декодера, возможно стоит сделать иначе?
```
For encoder-decoder hybrids, asymmetric RoPE configurations allow specialization: encoder uses high theta for bidirectional long-range understanding, decoder uses lower theta for autoregressive generation. This architectural separation—impossible in decoder-only models—enables simultaneous optimization for understanding and generation.
```

MUON не работает, или работает не так как задумано. Возможно из-за tie-embeddings

FA2 реализация для t5 сделана не очень красиво (из-за этого поддерживается только bf16\fp16), надо исправлять, но судя по всему работает.

Дистиляция c decoder-only? https://arxiv.org/pdf/2501.16273

gemma-t5 теоретически выглядит интересно, но не работает (или работает не так как задумано)