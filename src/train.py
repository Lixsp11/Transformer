import time
import sacrebleu as sb
import torch.utils.data
from utils import *
import MyTransformer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def test_bleu(_src: torch.Tensor, _tgt_s: torch.Tensor, _tgt_e: torch.Tensor) -> None:
    assert _src.ndim == _tgt_s.ndim == _tgt_e.ndim == 2, 'Input shape should be (1, seq)'
    global model, dataset, writer

    # Remove padding
    src_length, tgt_length = torch.count_nonzero(_src), torch.count_nonzero(_tgt_e)
    _src, _tgt_s, _tgt_e = _src[:, :src_length], _tgt_s[:, :tgt_length], _tgt_e[:, :tgt_length]

    model.eval()
    _t = model.forward(_src, _tgt_s).argmax(-1)
    _s = model.predict(_src, _tgt_s).argmax(-1)
    model.train()

    ground_truth = [dataset.vector2seq(_tgt_e[0])]
    _t = [[dataset.vector2seq(_t[0])]]
    _s = [[dataset.vector2seq(_s[0])]]
    _t_bleu = sb.corpus_bleu(ground_truth, _t).score
    _s_bleu = sb.corpus_bleu(ground_truth, _s).score

    print(f"g:{ground_truth}")
    print(f"t:{_t}, bleu={_t_bleu:2.4}")
    print(f"s:{_s}, bleu={_s_bleu:2.4}")
    writer.add_scalar('BLEU/TF', _t_bleu, global_step=iteration)
    writer.add_scalar('BLEU/SR', _s_bleu, global_step=iteration)


@torch.no_grad()
def log_data():
    global tgt_hat, tgt_e, epoch, iteration, optim, loss, time_
    tgt_hat = tgt_hat.argmax(-1)
    acc = ((tgt_hat == tgt_e) == (tgt_hat != 0)).sum() / (tgt_e != 0).sum()  # Without cal padding
    log = f"Epoch={epoch:03}, Iteration={iteration:06}, Lr={optim.lr:1.6f}, " \
          f"Loss={loss.detach().item():6.5f}, Acc={acc.item():1.2f}, " \
          f"Time={time.time() - time_:2.4f}"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('CELoss', loss.detach().item(), global_step=iteration)
    writer.add_scalar('Accuracy', acc.item(), global_step=iteration)
    writer.add_scalar('LRate', optim.lr, global_step=iteration)
    time_ = time.time()


if __name__ == '__main__':
    print(torch.__version__)

    writer = SummaryWriter('../log/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = WMT16(src_path='../data/train.tok.clean.bpe.32000.de',
                    tgt_path='../data/train.tok.clean.bpe.32000.en',
                    vocab_path='../data/vocab.bpe.31500', split_type='train')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, pin_memory=True,
                                             shuffle=True, collate_fn=collate_padding)
    model = MyTransformer.Transformer(num_encoder_layers=6, num_decoder_layers=6, d_model=512,
                                      n_head=8, vocab=len(dataset.index2word), encoder_d_ff=2048,
                                      decoder_d_ff=2048).to(device)
    Loss = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(device)
    scaler = GradScaler()
    optim = TransformerOptimizer(model.parameters(), d_model=512)

    iteration, epoch, time_ = 0, 0, time.time()
    while iteration < 1e5:
        epoch += 1
        for src, (tgt_s, tgt_e) in dataloader:
            iteration += 1

            with autocast():
                src, tgt_s, tgt_e = src.to(device), tgt_s.to(device), tgt_e.to(device)
                tgt_hat = model(src, tgt_s)
                loss = Loss(tgt_hat.reshape(-1, tgt_hat.shape[-1]), tgt_e.reshape(-1, ))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            if iteration % 100 == 0:
                log_data()
            if iteration % 5000 == 0:
                test_bleu(src[:1], tgt_s[:1], tgt_e[:1])
                torch.save(model.state_dict(), 'Transformer-re.pt')
