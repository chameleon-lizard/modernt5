from transformers import EncoderDecoderModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('deepvk/RuModernBERT-small')
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("deepvk/RuModernBERT-small", "gpt2")
print(bert2bert.num_parameters())

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 8192
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

bert2bert.save_pretrained('encdec')
tokenizer = AutoTokenizer.from_pretrained('deepvk/RuModernBERT-small')
