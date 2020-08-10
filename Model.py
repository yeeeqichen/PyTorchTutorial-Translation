import torch
from Config import config
import random
import numpy


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.enc_dict_size, config.enc_embed_size)
        self.rnn = torch.nn.GRU(config.enc_embed_size, config.enc_hidden_size, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(config.enc_hidden_size * 2, config.dec_hidden_size)
        self.dropout = torch.nn.Dropout(config.enc_dropout)

    def forward(self, enc_inputs):
        # print(enc_inputs)
        embed = self.dropout(self.embedding(enc_inputs))
        # print(embed.shape)
        # print(embed)
        outputs, hidden = self.rnn(embed)
        # print(outputs.shape, hidden.shape)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # print(hidden.shape)
        return outputs, hidden


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Linear(config.enc_hidden_size * 2 + config.dec_hidden_size, config.attn_dim)

    def forward(self, encode_outputs, decode_hidden):
        # print(encode_outputs.shape, decode_hidden.shape)
        seq_len = encode_outputs.shape[1]
        repeated_decode_hidden = decode_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # encode_outputs = encode_outputs.permute(1, 0, 2)
        weights = torch.tanh(self.attention(torch.cat((encode_outputs, repeated_decode_hidden), dim=2)))
        weights = torch.sum(weights, dim=2)
        return torch.nn.functional.softmax(weights, dim=1)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(config.dec_dict_size, config.dec_embed_size)
        self.rnn = torch.nn.GRU(config.enc_hidden_size * 2 + config.dec_embed_size, config.dec_hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(config.enc_hidden_size * 2 + config.dec_hidden_size + config.dec_embed_size,
                                  config.dec_dict_size)
        self.dropout = torch.nn.Dropout(config.dec_dropout)
        self.att = Attention()

    @staticmethod
    def _get_weighted_outputs(enc_outputs, weights):
        # print(weights.shape)
        weights = weights.unsqueeze(1)
        # enc_outputs = enc_outputs.premute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(weights, enc_outputs)
        # print(weighted_encoder_rep.shape)
        # weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self, enc_outputs, dec_input, dec_hidden):
        dec_input = dec_input.unsqueeze(1)
        # print(dec_input)
        embed = self.dropout(self.embed(dec_input))
        att_weights = self.att(enc_outputs, dec_hidden)
        att_enc_outs = self._get_weighted_outputs(enc_outputs, att_weights)
        # print(embed.shape)
        rnn_input = torch.cat((att_enc_outs, embed), dim=2)
        # print(rnn_input.shape)
        output, hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(0))
        # print(output.shape)
        output = self.fc(torch.cat((att_enc_outs, output, embed), dim=2))
        # print(output.shape)
        return output.squeeze(1), hidden.squeeze(1)


class Seq2Seq(torch.nn.Module):
    """
    Seq2Seq 模型
    输入:(batch, sequence, features)
    输出:(sequence, batch, dic_size)
    对 dic_size 维度取max即得到预测结果（词典中的索引）
    注意：需要作padding
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, src, tgt, teacher_force_ratio):
        length = tgt.shape[1]
        enc_outputs, enc_hidden = self.encoder(src)
        outputs = torch.zeros(length, config.batch_size, config.dec_dict_size).to(self.device)
        output = tgt[:, 0]
        for i in range(1, length):
            output, hidden = self.decoder(enc_outputs, output, enc_hidden)
            outputs[i] = output
            teacher_force = random.random() < teacher_force_ratio
            top1 = output.max(1)[1]
            output = (tgt[:, i] if teacher_force else top1)
        return outputs


model = Seq2Seq()
src = torch.LongTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
tgt = torch.LongTensor([[0, 1, 2], [0, 2, 3], [0, 1, 3]])
print(model)
print(model(src, tgt, 0))
