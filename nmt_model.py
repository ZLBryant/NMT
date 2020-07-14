from torch import nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import random

class NMT(nn.Module):
    def __init__(self, args, vocab):
        super(NMT, self).__init__()
        self.args = args
        self.vocab = vocab
        self.src_emb = nn.Embedding(len(vocab.src), args.emb_dim, padding_idx=vocab.src.word2id["<pad>"])
        self.tgt_emb = nn.Embedding(len(vocab.tgt), args.emb_dim, padding_idx=vocab.tgt.word2id["<pad>"])
        self.encoder = nn.LSTM(input_size=args.emb_dim, hidden_size=args.hidden_dim,
                               bidirectional=args.encoder_bidirectional,
                               num_layers=args.encoder_layer_num, batch_first=args.batch_first, bias=True)
        self.decoder = nn.LSTM(input_size=args.emb_dim + args.hidden_dim, hidden_size=args.hidden_dim,
                               bidirectional=False, num_layers=args.decoder_layer_num, batch_first=args.batch_first, bias=True)
        encoder_hidden_dim = args.hidden_dim * 2 if args.encoder_bidirectional else args.hidden_dim
        self.attention_layer = nn.Linear(encoder_hidden_dim, args.hidden_dim)
        self.c_project = nn.Linear(encoder_hidden_dim * args.encoder_layer_num, args.hidden_dim)
        self.h_project = nn.Linear(encoder_hidden_dim * args.encoder_layer_num, args.hidden_dim)
        self.bridge_project = nn.Linear(args.hidden_dim * 3, args.hidden_dim)
        self.predict = nn.Linear(args.hidden_dim, len(vocab.tgt))
        self.dropout = nn.Dropout(args.dropout)
        self.model_init()

    def model_init(self):
        for p in self.parameters():
            p.data.uniform_(-self.args.uniform_init, self.args.uniform_init)
        return

    def generate_mask(self, sent_lens):
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        src_mask = torch.ones(batch_size, max_len).to(self.args.device)
        for i in range(batch_size):
            src_mask[i, :sent_lens[i]] = 0
        return src_mask

    def encode(self, src_pad, src_lens):
        src_pad_emb = self.src_emb(src_pad)
        enc_input = pack_padded_sequence(src_pad_emb, src_lens, batch_first=self.args.batch_first)
        enc_hiddens, enc_state = self.encoder(enc_input)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=self.args.batch_first)
        num = self.args.encoder_layer_num * 2 if self.args.encoder_bidirectional else self.args.encoder_layer_num
        dec_hidden = [enc_state[0][i, :, :] for i in range(num)]#每个元素为batch_size*hidden_size
        dec_hidden = torch.cat(dec_hidden, dim=-1)#看一下是三维还是二维
        dec_hidden = self.h_project(dec_hidden).unsqueeze(0).repeat(self.args.decoder_layer_num, 1, 1)
        dec_cell = [enc_state[1][i, :, :] for i in range(num)]
        dec_cell = torch.cat(dec_cell, dim=-1)  # 看一下是三维还是二维
        dec_cell = self.c_project(dec_cell).unsqueeze(0).repeat(self.args.decoder_layer_num, 1, 1)
        return enc_hiddens, (dec_hidden, dec_cell)

    def step(self, input, enc_hiddens, enc_attn_project, dec_state, enc_mask=None):
        dec_final_hidden, dec_state = self.decoder(input, dec_state)
        attn = torch.matmul(enc_attn_project, dec_final_hidden.permute(0, 2, 1)).squeeze(-1)#enc_attn_project:batch_size*seq_len*hidden_size,dec_final_hidden:batch_size*1*hidden_size
        if enc_mask != None:
            attn.data.masked_fill_(enc_mask.bool(), -float('inf'))
        attn = F.softmax(input=attn, dim=-1)
        context = torch.matmul(attn.unsqueeze(1), enc_hiddens).squeeze(1)
        context = torch.cat((context, dec_final_hidden.squeeze(1)), dim=-1)
        context = self.bridge_project(context)
        context = self.dropout(torch.tanh(context))
        output = torch.tanh(context)
        return output, dec_state

    def decode(self, enc_hiddens, tgt_pad, dec_state, teaching_force, enc_mask=None):
        tgt_pad_emb = self.tgt_emb(tgt_pad)
        tgt_input = tgt_pad_emb[:, :-1]
        prev = torch.zeros(enc_hiddens.shape[0], 1, self.args.hidden_dim).to(self.args.device)
        enc_attn_project = self.attention_layer(enc_hiddens)
        logits = []
        logit = None
        for input_unit in torch.split(tgt_input, 1, dim=1):
            if random.random() > teaching_force and logit != None:
                greed_ids = torch.argmax(logit, dim=-1)
                greedy_pred = self.tgt_emb(greed_ids)
                input_unit = greedy_pred.unsqueeze(1)
            input_unit = torch.cat((input_unit, prev), dim=-1)
            output, dec_state = self.step(input_unit, enc_hiddens, enc_attn_project, dec_state, enc_mask)
            prev = output.unsqueeze(1)
            logit = self.predict(output)
            logits.append(logit.unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        return logits

    def forward(self, src_sents, tgt_sents, teaching_force=1):
        src_lens = [len(sent) for sent in src_sents]
        src_mask = self.generate_mask(src_lens)
        src_pad = self.vocab.src.to_input_tensor(src_sents).to(self.args.device)
        tgt_pad = self.vocab.tgt.to_input_tensor(tgt_sents).to(self.args.device)
        enc_hiddens, dec_init_state = self.encode(src_pad, src_lens)
        logits = self.decode(enc_hiddens, tgt_pad, dec_init_state, teaching_force, src_mask)
        log_logit = F.log_softmax(logits, dim=-1)
        tgt = tgt_pad[:, 1:]
        tgt_mask = tgt != self.vocab.tgt["<pad>"]
        loss = log_logit.gather(index=tgt.unsqueeze(-1), dim=-1).squeeze(-1) * tgt_mask
        loss = loss.sum(-1)
        if self.training:
            return loss
        else:
            predict_sentences_id = torch.argmax(logits, dim=-1).squeeze(dim=-1)
            #bos = torch.ones(predict_sentences_id.shape[0], 1).to(self.args.device) * self.vocab.tgt["<s>"]
            #predict_sentences_id = torch.cat((bos, predict_sentences_id), dim=0)
            predict_sentences = []
            for sent in torch.split(predict_sentences_id, 1, dim=0):
                predict_sentences.append(self.vocab.tgt.indices2words(sent.squeeze(0).cpu().numpy().tolist()))
            return loss, predict_sentences

    def translate(self, src_sent):
        src_sent_id = self.vocab.src.to_input_tensor([src_sent]).to(self.args.device)
        enc_hiddens, dec_state = self.encode(src_sent_id, [len(src_sent)])
        prev = torch.zeros(1, self.args.hidden_dim).to(self.args.device)
        enc_attn_project = self.attention_layer(enc_hiddens)
        beam_sents = [[self.vocab.tgt["<s>"]]]
        beam_scores = torch.Tensor([0]).to(self.args.device)
        cur_len = 0
        complete_beams = []
        while len(complete_beams) < self.args.beam_size and cur_len < self.args.max_len:
            cur_len += 1
            cur_beam_num = len(beam_sents)
            enc_hiddens_repeat = enc_hiddens.repeat(cur_beam_num, 1, 1)
            enc_attn_project_repeat = enc_attn_project.repeat(cur_beam_num, 1, 1)
            input = torch.Tensor([beam_sent[-1] for beam_sent in beam_sents]).long().to(self.args.device)
            input = self.tgt_emb(input)
            input = torch.cat((input, prev), dim=-1)
            output, dec_state = self.step(input.unsqueeze(1), enc_hiddens_repeat, enc_attn_project_repeat, dec_state)
            log_prob = F.log_softmax(self.predict(output), dim=-1)
            new_beam_scores_candidate = (beam_scores.unsqueeze(1).expand_as(log_prob) + log_prob).view(-1)
            topk_scores, topk_ids = torch.topk(new_beam_scores_candidate, k=self.args.beam_size - len(complete_beams))
            beam_ids = topk_ids / len(self.vocab.tgt)
            word_ids = topk_ids % len(self.vocab.tgt)
            new_beam_sents = []
            new_beam_scores = []
            new_beam_ids = []
            for score, beam_id, word_id in zip(topk_scores, beam_ids, word_ids):
                score = score.item()
                beam_id = beam_id.item()
                word_id = word_id.item()
                word = self.vocab.tgt.id2word[word_id]
                sentence = beam_sents[beam_id] + [word_id]
                if word == '</s>':
                    score = score / len(sentence)
                    complete_beam = {}
                    sentence = self.vocab.tgt.indices2words(sentence)
                    complete_beam["sentence"] = sentence[1:]
                    complete_beam["score"] = score
                    complete_beams.append(complete_beam)
                else:
                    new_beam_sents.append(sentence)
                    new_beam_scores.append(score)
                    new_beam_ids.append(beam_id)
            beam_sents = new_beam_sents
            beam_scores = torch.Tensor(new_beam_scores).to(self.args.device)
            prev = output[new_beam_ids, :]
            dec_state = (dec_state[0][:, new_beam_ids, :], dec_state[1][:, new_beam_ids, :])

        if len(complete_beams) == 0:
            complete_beam = {}
            sentence = self.vocab.tgt.indices2words(beam_sents[0])#注意topk使得最优选项在0处
            complete_beam["sentence"] = sentence[1:]
            complete_beam["score"] = beam_scores[0]
            complete_beams.append(complete_beam)
        complete_beams.sort(key=lambda beam: beam["score"], reverse=True)
        return complete_beams
