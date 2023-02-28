from torch import nn
from TorchCRF import CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, pad_index,batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.pad_idx = pad_index
        self.batch_size = batch_size
        
        self.word_embeds = nn.Embedding(vocab_size,embedding_dim,padding_idx=self.pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 1, bidirectional = True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
        self.crf = CRF(self.tagset_size)
    
   
    def forward(self, sentence, tags, mask): 
        embeds = self.word_embeds(sentence.long())
        lstm_out = self.lstm(embeds) 
        lstm_feats = self.hidden2tag(lstm_out[0])   
        loss = -self.crf.forward(lstm_feats,tags,mask)  
        return loss