class Config:
    def __init__(self):
        self.enc_dict_size = 1000
        self.enc_embed_size = 128
        self.dec_dict_size = 1000
        self.dec_embed_size = 128
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.enc_hidden_size = 256
        self.dec_hidden_size = 256
        self.enc_dropout = 0.2
        self.dec_dropout = 0.2
        self.batch_size = 3
        self.max_len = 20
        self.attn_dim = 8
        pass


config = Config()
