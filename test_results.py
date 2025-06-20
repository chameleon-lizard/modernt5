import argparse
import transformers

import modeling_modernt5


def generate(text, model, tokenizer):
    print(text, end=' | ')
    input_ids = tokenizer('[BOS]' + text, return_tensors='pt')
    res = model.generate(**input_ids.to('cuda'))
    print(tokenizer.decode(res[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a pretrained model.")
    parser.add_argument("--model_path", type=str, default="results",
                        help="Path to the directory containing the trained model.")
    args = parser.parse_args()

    model = modeling_modernt5.ModernT5ForConditionalGeneration.from_pretrained(args.model_path).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained('deepvk/RuModernBERT-base')

    gen = lambda text: generate(text, model, tokenizer)

    gen('I love my mother very ')
    gen('Я очень ')
    gen('London is the ')
    gen('Москва - это ')
