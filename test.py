import argparse
import torch
from utils import read_corpus, read_en_cn_corpus, batch_iter
from vocab import Vocab
from nmt_model import NMT
from torch.optim import Adam
import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def model_train(model):
    if model.args.en_cn:
        train_src, train_tgt = read_en_cn_corpus(model.args.train_data, source=model.args.source_language)
        dev_src, dev_tgt = read_en_cn_corpus(model.args.dev_data, source=model.args.source_language)
    elif model.args.en_es:
        train_src = read_corpus(model.args.train_src_data, False)
        train_tgt = read_corpus(model.args.train_tgt_data, True)
        dev_src = read_corpus(model.args.dev_src_data, False)
        dev_tgt = read_corpus(model.args.dev_tgt_data, True)
    else:
        print("invalid input")
        exit(0)
    train_data = list(zip(train_src, train_tgt))
    dev_data = list(zip(dev_src, dev_tgt))

    batch_size = model.args.train_batch_size
    lr = model.args.lr
    lr_decay = model.args.lr_decay
    log_iter_num = model.args.log_iter_num
    optimizer = Adam(model.parameters(), lr=lr)
    clip_grad = args.clip_grad
    eval_iter_num = args.eval_iter_num
    epoch = model.args.max_epoch
    model_save_path = model.args.model_save_path

    patience =exit_count = cur_iter = 0
    log_loss_sum = log_sample_num = 0
    start_time = time.time()
    best_ppl = float('inf')
    for cur_epoch in range(epoch):
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=batch_size, shuffle=True):
            teaching_force = 1 - cur_epoch / epoch
            #teaching_force = 1
            cur_iter += 1
            cur_batch_size = len(src_sents)
            loss = -model(src_sents, tgt_sents, teaching_force).sum()
            log_loss_sum += loss.item()
            log_sample_num += cur_batch_size

            loss = loss / cur_batch_size
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            if cur_iter % log_iter_num == 0:
                print("log:  epoch %d, cur_iter %d, loss %2f, time elapsed %2f second" % (
                    cur_epoch, cur_iter, log_loss_sum / log_sample_num, time.time() - start_time))
                log_loss_sum = log_sample_num = 0

            if cur_iter % eval_iter_num == 0:
                loss, ppl, bleu = model_eval(model, dev_data)
                print("eval:  epoch %d, cur_iter %d, loss %2f, ppl %2f, bleu %2f, time elapsed %2f second" % (
                    cur_epoch, cur_iter, loss, ppl, bleu, time.time() - start_time))
                if best_ppl > ppl:
                    patience = 0
                    best_ppl = ppl
                    torch.save(model, model.args.model_save_path)
                else:
                    patience += 1
                    if patience == model.args.patience:
                        exit_count += 1
                        if exit_count == model.args.exit_threshold:
                            exit(0)
                        lr = lr * lr_decay
                        model = torch.load(model_save_path)
                        model.to(model.args.device)
                        optimizer = Adam(model.parameters(), lr=lr)
                        patience = 0

def eval_bleu(predict_sents, tgt_sents):
    bleu_scores = []
    sents = zip(predict_sents, tgt_sents)
    for predict_sent, tgt_sent in sents:
        bleu_score = sentence_bleu([tgt_sent], predict_sent)
        bleu_scores.append(bleu_score)
    bleu_sum = sum(bleu_scores)
    return bleu_sum

def model_eval(model, eval_data):
    model.eval()
    batch_size = model.args.eval_batch_size
    shuffle = model.args.shuffle
    loss_sum = predict_word_num = bleu_sum = predict_sent_num = 0
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(eval_data, batch_size, shuffle=shuffle):
            loss, predict_sents = model(src_sents, tgt_sents)
            loss = -loss.sum()
            loss_sum += loss.item()
            predict_word_num += sum([len(sent) for sent in tgt_sents])
            tgt_sents = [sent[1: -1] for sent in tgt_sents]
            bleu_sum += eval_bleu(predict_sents, tgt_sents)
            predict_sent_num += len(src_sents)
    model.train()
    loss = loss_sum / predict_sent_num
    ppl = np.exp(loss_sum / predict_word_num)
    bleu = bleu_sum / predict_sent_num
    return loss, ppl, bleu

def translate(model):
    if model.args.en_cn:
        test_src, test_tgt = read_en_cn_corpus(model.args.test_data, source=model.args.source_language)
    elif model.args.en_es:
        test_src = read_corpus(model.args.test_src_data, False)
        test_tgt = read_corpus(model.args.test_tgt_data, True)
    else:
        print("invalid input")
        exit(0)
    train_data = list(zip(test_src, test_tgt))
    with open(args.translate_result_save_path, 'w') as f:
        for src_sent, tgt_sent in batch_iter(train_data, 1, shuffle=False):
            tgt_predict = model.translate(src_sent[0])
            tgt_predict = ' '.join(tgt_predict[0]['sentence'])
            tgt_sent = tgt_sent[0]
            tgt = ' '.join(tgt_sent[1: -1])
            line = "translate: " + tgt_predict + " " * 5 + "target: " + tgt + '\r\n'
            f.write(line)
    f.close()

def args_init():
    parser = argparse.ArgumentParser(description="Neural MT")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--en_cn", action="store_true")
    parser.add_argument("--source_language", type=str, default='en')
    parser.add_argument("--train_data", type=str, default="dataset/en_cn_data/train.txt", help="train file")
    parser.add_argument("--dev_data", type=str, default="dataset/en_cn_data/dev.txt", help="dev file")
    parser.add_argument("--test_data", type=str, default="dataset/en_cn_data/test.txt", help="test file")
    parser.add_argument("--en_es", action="store_true")
    parser.add_argument("--train_src_data", type=str, default="dataset/en_es_data/train.es", help="train src file")
    parser.add_argument("--dev_src_data", type=str, default="dataset/en_es_data/dev.es", help="dev src file")
    parser.add_argument("--test_src_data", type=str, default="dataset/en_es_data/test.es", help="test src file")
    parser.add_argument("--train_tgt_data", type=str, default="dataset/en_es_data/train.en", help="train tgt file")
    parser.add_argument("--dev_tgt_data", type=str, default="dataset/en_es_data/dev.en", help="dev tgt file")
    parser.add_argument("--test_tgt_data", type=str, default="dataset/en_es_data/test.en", help="test tgt file")

    parser.add_argument("--vocab_file_path", type=str, default='vocab.json')
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--batch_first", action="store_false")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_layer_num", type=int, default=1)
    parser.add_argument("--encoder_bidirectional", type=bool, default=True)
    parser.add_argument("--decoder_layer_num", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--uniform_init", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--clip_grad", type=float, default=5.)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_iter_num", type=int, default=10)
    parser.add_argument("--eval_iter_num", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--exit_threshold", type=int, default=5)
    parser.add_argument("--model_save_path", type=str, default="output/model_es2en.pkl")
    parser.add_argument("--translate_result_save_path", type=str, default="output/translate_result.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=50)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_init()
    args.device = ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(args.seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args.train:
        vocab = Vocab.load(args.vocab_file_path)
        model = NMT(args, vocab).to(args.device)
        model = model_train(model)
    else:
        model = torch.load(args.model_save_path)
        model = model.to(args.device)
        model.args = args
        translate(model)