import codecs


def validation(vocab_path: str) -> None:
    global src_path, tgt_path, dataset_size
    with codecs.open(vocab_path, 'rb', 'utf-8') as f:
        vocab = f.read().split('\n')
    print(len(vocab))
    with codecs.open(src_path, 'rb', 'utf-8') as f:
        file = f.read().split('\n')
        for i, seq in enumerate(file):
            for word in seq.split(' '):
                assert word in vocab, f'?{word}'
            print(i)
            if i == dataset_size:
                break
    with codecs.open(tgt_path, 'rb', 'utf-8') as f:
        file = f.read().split('\n')
        for i, seq in enumerate(file):
            for word in seq.split(' '):
                assert word in vocab, f'?{word}'
            print(i)
            if i == dataset_size:
                break
    print('done')


def partition(vocab_path: str) -> None:
    global src_path, tgt_path, dataset_size
    voacb = set()

    with codecs.open(src_path, 'rb', 'utf-8') as f:
        file = f.read().split('\n')
        for i, seq in enumerate(file):
            for word in seq.split(' '):
                voacb.add(word)
            print(i)
            if i == dataset_size:
                break
    with codecs.open(tgt_path, 'rb', 'utf-8') as f:
        file = f.read().split('\n')
        for i, seq in enumerate(file):
            for word in seq.split(' '):
                voacb.add(word)
            print(i)
            if i == dataset_size:
                break

    with codecs.open('vocab_path', 'wb', 'utf-8') as f:
        for word in voacb:
            f.write(word + '\n')


if __name__ == '__main__':
    src_path = '../data/train.tok.clean.bpe.32000.de'
    tgt_path = '../data/train.tok.clean.bpe.32000.en'
    dataset_size = 1000000 + 10000

    partition('../data/vocab.bpe.31500')
    validation('../data/vocab.bpe.31500')


