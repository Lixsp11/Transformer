# Transformer
Implementation of paper "Attention is all you need"

### Requirement

- python 3.8+
- pytorch 1.10+ (For label smoothing supported in CrossEntropyLoss)
- sacrebleu

### Dataset

We used [WMT'16 EN-DE Dataset](https://google.github.io/seq2seq/data/), and preprocessed using  [wmt16_en_de.sh](https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh) data generation script provided by Google. The script downloads the data, tokenizes it using the [Moses Tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl), cleans the training data, and learns a vocabulary of ~32,000 subword units. We use the following three files from the script's generated content for training:

| Filename                       | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `train.tok.clean.bpe.32000.en` | The English training data, one sentence per line, processed using BPE. |
| `train.tok.clean.bpe.32000.de` | The German training data, one sentence per line, processed using BPE. |
| `vocab.bpe.32000`              | The full vocabulary used in the training data, one token per line. |

### Train

```bash
python train.py
```

