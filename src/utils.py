import torch
import codecs
import torch.utils.data
from typing import Tuple, Union, Iterable, Callable, Any, List


class TransformerOptimizer(torch.optim.Adam):
    """Optimizer for Transformer based on Adam optimizer.
    lrate = d_model^0.5 * min(step_num^-0.5, step_num*warmup_steps^-1.5

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups (required).
        d_model: The number of expected features in the input (required).
        warmup_steps: Increase learning rate linearly for the first warmup_steps (default=4000).
        betas: Coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.98)).
        eps: Term added to the denominator to improve numerical stability (default: 1e-9).
    """

    def __init__(self, params: Union[Iterable[torch.Tensor], Iterable[dict]], d_model: int,
                 warmup_steps: int = 4000, betas: Tuple[float, float] = (0.9, 0.98),
                 eps: float = 1e-9) -> None:
        super(TransformerOptimizer, self).__init__(params, betas=betas, eps=eps)
        self._step_num = 0
        self._d_model = d_model
        self._warmup_steps = warmup_steps
        self.lr = None
        self.lrate = lambda x: self._d_model ** -0.5 * min(x ** -0.5,
                                                           x * self._warmup_steps ** -1.5)

    def step(self, closure: Callable = None) -> None:
        self._step_num += 1
        self.lr = self.lrate(self._step_num)
        for param_group in self.param_groups:
            param_group['lr'] = self.lr
        super(TransformerOptimizer, self).step(closure)


class WMT16(torch.utils.data.Dataset):
    r"""WMT16 EN-DE dataset for natural language translation tasks.

    Args:
        src_path: Source language dataset path (required).
        tgt_path: Target language dataset path (required).
        vocab_path: Shared Dictionary path(required).
        split_type: Dataset split type: 'Train' or 'Eval' (required).
    """
    def __init__(self, src_path: str, tgt_path: str, vocab_path: str, split_type: str) -> None:
        super(WMT16, self).__init__()
        self.word2index = {}
        self.index2word = [u'[PAD]', u'[SOS]', u'[EOS]']
        self.src_seqs = []
        self.tgt_seqs = []

        assert split_type in ('train', 'eval'), f'Unknown dataset type {split_type}.'
        self.dataset_size = {'train': (0, 1000000), 'eval': (1000000, 1010000)}

        with codecs.open(vocab_path, 'rb', 'utf-8') as f:
            file = f.read().split('\n')
            self.index2word.extend(file)
            for index, word in enumerate(self.index2word):
                self.word2index[word] = index

        with codecs.open(src_path, 'rb', 'utf-8') as f:
            file = f.read().split('\n')
            for i, seq in enumerate(file):
                if self.dataset_size[split_type][0] <= i < self.dataset_size[split_type][1]:
                    self.src_seqs.append(self.seq2vector(seq.split(' ')))
                if i >= self.dataset_size[split_type][1]:
                    break

        with codecs.open(tgt_path, 'rb', 'utf-8') as f:
            file = f.read().split('\n')
            for i, seq in enumerate(file):
                if self.dataset_size[split_type][0] <= i < self.dataset_size[split_type][1]:
                    seq = seq.split(' ')
                    self.tgt_seqs.append(
                        (self.seq2vector([u'[SOS]'] + seq), self.seq2vector(seq + [u'[EOS]'])))
                if i >= self.dataset_size[split_type][1]:
                    break
        print(f'{split_type} dataset, size={len(self.src_seqs)}')

    def seq2vector(self, seq: List[str]) -> torch.Tensor:
        vector = []
        for word in seq:
            assert word in self.word2index.keys(), f"Can't find word {word} in vocab."
            vector.append(self.word2index[word])
        return torch.tensor(vector)

    def vector2seq(self, vector: torch.Tensor) -> str:
        assert vector.ndim == 1, f'Vector.ndim should equal to 1, found {vector.ndim}'
        seq = []
        for index in vector:
            assert 0 <= index < len(self.index2word), f"Can't find index {index} in vocab."
            seq.append(self.index2word[index])
        return ' '.join(seq)

    def __getitem__(self, item: Any) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.src_seqs[item], self.tgt_seqs[item]

    def __len__(self) -> int:
        return len(self.src_seqs)


def collate_padding(batch_data: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]) \
        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Pad the sentences for batch training. Used as collate_fn in dataloader."""
    srcs = []
    tgts_s = []
    tgts_e = []
    for src, (tgt_s, tgt_e) in batch_data:
        srcs.append(src)
        tgts_s.append(tgt_s)
        tgts_e.append(tgt_e)
    srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts_s = torch.nn.utils.rnn.pad_sequence(tgts_s, batch_first=True, padding_value=0)
    tgts_e = torch.nn.utils.rnn.pad_sequence(tgts_e, batch_first=True, padding_value=0)
    return srcs, (tgts_s, tgts_e)
