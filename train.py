from __future__ import division, print_function
from tqdm import tqdm
import torch

from utils import shell, init_weights, set_seed
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from get_args import setup, model_setup, clean_up
from dataloader import Dataset
from model import LSTMClassifier
from evaluate import Validator, Predictor


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train(model=None, train_dl=None, validator=None,
          tester=None, epochs=20, lr=0.001, log_every_n_examples=1,
          weight_decay=0):
    # opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                       lr=lr, momentum=0.9)
    opt = torch.optim.Adadelta(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1.0, rho=0.9,
        eps=1e-6, weight_decay=weight_decay)

    for epoch in tqdm(range(epochs)):
        if epoch - validator.best_epoch > 10:
            return

        pbar = train_dl
        total_loss = 0
        n_correct = 0
        cnt = 0

        model.train()

        for batch in pbar:
            batch_size = len(batch.tgt)

            loss, acc = model.loss_n_acc(batch.input, batch.tgt)
            total_loss += loss.item() * batch_size
            cnt += batch_size
            n_correct += acc

            opt.zero_grad()
            loss.backward()
            clip_gradient(model, 1)
            opt.step()

        model.eval()
        validator.evaluate(model, epoch)
        tester.evaluate(model, epoch)

        summ = {
            'Eval': '(e{:02d},train)'.format(epoch),
            'loss': total_loss / cnt,
            'acc': n_correct / cnt,
        }
        validator.write_summary(summ=summ)
        validator.write_summary(epoch=epoch)
        tester.write_summary(epoch)


def bookkeep(predictor, validator, tester, args, INPUT_field):
    tester.final_evaluate(predictor.model)

    predictor.pred_sent(INPUT_field)

    save_model_fname = validator.save_model_fname + '.e{:02d}.loss{:.4f}.torch'.format(
        validator.best_epoch, validator.best_loss)
    cmd = 'cp {} {}'.format(validator.save_model_fname, save_model_fname)
    shell(cmd)

    clean_up(args)


def run(args):
    set_seed(args.seed)

    dataset = Dataset(proc_id=0, data_dir='tmp2/',
                      train_fname='train.csv',
                      preprocessed=True, lower=True,
                      vocab_max_size=100000, emb_dim=100,
                      save_vocab_fname=args.save_vocab_fname, verbose=True, )
    train_dl, valid_dl, test_dl = \
        dataset.get_dataloader(proc_id=0, batch_size=args.batch_size)

    validator = Validator(dataloader=valid_dl, save_dir=args.save_dir,
                          save_log_fname=args.save_log_fname,
                          save_model_fname=args.save_model_fname,
                          valid_or_test='valid',
                          vocab_itos=dataset.INPUT.vocab.itos,
                          label_itos=dataset.TGT.vocab.itos)
    tester = Validator(dataloader=test_dl, save_log_fname=args.save_log_fname,
                       save_dir=args.save_dir, valid_or_test='test',
                       vocab_itos=dataset.INPUT.vocab.itos,
                       label_itos=dataset.TGT.vocab.itos)
    predictor = Predictor(args.save_vocab_fname)

    if args.load_model:
        predictor.use_pretrained_model(args.load_model)
        import pdb;
        pdb.set_trace()

        predictor.pred_sent(dataset.INPUT)
        tester.final_evaluate(predictor.model)

        return

    model = LSTMClassifier(emb_vectors=dataset.INPUT.vocab.vectors,
                           emb_dropout=args.emb_dropout,
                           lstm_dim=args.lstm_dim,
                           lstm_n_layer=args.lstm_n_layer,
                           lstm_dropout=args.lstm_dropout,
                           lstm_combine=args.lstm_combine,
                           linear_dropout=args.linear_dropout,
                           n_linear=args.n_linear,
                           n_classes=len(dataset.TGT.vocab))
    if args.init_xavier: model.apply(init_weights)
    args = model_setup(0, model, args)

    train(model=model, train_dl=train_dl,
          validator=validator, tester=tester, epochs=args.epochs, lr=args.lr,
          weight_decay=args.weight_decay)

    predictor.use_pretrained_model(args.save_model_fname)
    bookkeep(predictor, validator, tester, args, dataset.INPUT)


def main():
    args = setup()
    run(args)

if __name__ == '__main__':
    main()
