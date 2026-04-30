import torch
import torch.nn as nn
from torch.nn import functional as F

def l2norm(x, dim=-1):
    return x / x.norm(2, dim=dim, keepdim=True).clamp(min=1e-6)

def cosine_sim(im, s):
    return im.mm(s.t())

class ResidualLayer(nn.Module): # from kim
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class CharProbRNN(nn.Module):
    def __init__(self, num_chars, state_dim=256, hidden_size=256, num_layers=4, dropout=0.):
        super(CharProbRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.top_fc = nn.Linear(hidden_size, num_chars)
        self.char_embs = nn.Embedding(num_chars, hidden_size)

        # self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.ReLU())
        self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size*num_layers), nn.ReLU())

        torch.nn.init.kaiming_normal_(self.char_embs.weight.data)

    def forward(self, chars, cat_embs, set_grammar=True):
        char_embs, cat_embs = self.prep_input(chars, cat_embs)
        Hs = []
        lens = 0
        for cat_tensor in cat_embs: # each cat at one time
            # for simple RNNs
            # # cat_tensor is batch, dim
            # cat_tensor = cat_tensor.unsqueeze(0).expand(self.num_layers, -1, -1)
            # cat_tensor = self.cat_emb_expansion(cat_tensor)
            # all_hs, _ = self.rnn.forward(char_embs, cat_tensor)
            # all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            # Hs.append(all_hs[0].transpose(0,1))
            # lens = all_hs[1]

            # for LSTMs with 3d linears
            cat_tensor = self.cat_emb_expansion(cat_tensor) # batch, hidden*numlayers
            cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
            cat_tensor = cat_tensor.permute(2, 0, 1)
            h0_tensor = torch.zeros_like(cat_tensor)
            all_hs, _ = self.rnn.forward(char_embs, (h0_tensor, cat_tensor))
            all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            Hs.append(all_hs[0].transpose(0,1))
            lens = all_hs[1]

        Hses = torch.stack(Hs, 0)
        # Hses = nn.functional.relu(Hses)
        scores = self.top_fc.forward(Hses) # cats, batch, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        total_logprobs = []

        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :] # cats, (batch_scalar), num_chars_in_word, num_chars
            sent_id = idx // len(chars[0])
            word_id = idx % len(chars[0])
            targets = chars[sent_id][word_id][1:]
            this_word_logprobs = this_word_logprobs[:, range(this_word_logprobs.shape[1]), targets]  # cats, num_chars_in_word
            total_logprobs.append(this_word_logprobs.sum(-1)) # cats
        total_logprobs = torch.stack(total_logprobs, dim=0) # batch, cats
        total_logprobs = total_logprobs.reshape(len(chars), -1, total_logprobs.shape[1]) # sentbatch, wordbatch, cats
        # total_logprobs = total_logprobs.transpose(0, 1) # wordbatch, sentbatch, cats
        return total_logprobs

    def prep_input(self, chars, cat_embs):
        # cat_embs is num_cat, cat_dim
        # chars is num_words, word/char_tensor
        embeddings = []
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                embeddings.append(self.char_embs.forward(chars[i][j][:-1])) # no word end token
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False) # len, batch, embs
        expanded_cat_embs = cat_embs.unsqueeze(1).expand(-1, packed_char_embs.data.size(0), -1) # numcat,batch, catdim

        return packed_char_embs, expanded_cat_embs


class CharProbRNNVarLen(CharProbRNN):
    """
    CharProbRNN variant that preserves the original scoring behavior while
    allowing different numbers of words per sentence in the same batch.
    """

    @staticmethod
    def _flatten_words(chars):
        flat_inputs = []
        flat_targets = []
        sent_word_counts = []
        for sent in chars:
            sent_word_counts.append(len(sent))
            for word in sent:
                flat_inputs.append(word[:-1])
                flat_targets.append(word[1:])
        return flat_inputs, flat_targets, sent_word_counts

    def forward(self, chars, cat_embs, set_grammar=True):
        flat_inputs, flat_targets, sent_word_counts = self._flatten_words(chars)
        batch_size = len(chars)
        num_cats = cat_embs.shape[0]
        max_words = max(sent_word_counts, default=0)

        if not flat_inputs:
            return torch.empty(
                (batch_size, max_words, num_cats),
                device=cat_embs.device,
                dtype=self.char_embs.weight.dtype,
            )

        embeddings = [self.char_embs.forward(word) for word in flat_inputs]
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False)
        expanded_cat_embs = cat_embs.unsqueeze(1).expand(-1, len(flat_inputs), -1)

        Hs = []
        lens = None
        for cat_tensor in expanded_cat_embs:  # each cat at one time
            cat_tensor = self.cat_emb_expansion(cat_tensor)  # words, hidden*numlayers
            cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
            cat_tensor = cat_tensor.permute(2, 0, 1).contiguous()
            h0_tensor = torch.zeros_like(cat_tensor)
            all_hs, _ = self.rnn.forward(packed_char_embs, (h0_tensor, cat_tensor))
            all_hs, lens = nn.utils.rnn.pad_packed_sequence(all_hs)  # len, words, embs
            Hs.append(all_hs.transpose(0, 1))

        Hses = torch.stack(Hs, 0)
        scores = self.top_fc.forward(Hses)  # cats, words, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)

        flat_logprobs = []
        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :]
            targets = flat_targets[idx]
            target_positions = torch.arange(length, device=this_word_logprobs.device)
            this_word_logprobs = this_word_logprobs[:, target_positions, targets]
            flat_logprobs.append(this_word_logprobs.sum(-1))

        flat_logprobs = torch.stack(flat_logprobs, dim=0)
        total_logprobs = flat_logprobs.new_zeros((batch_size, max_words, flat_logprobs.shape[1]))

        flat_index = 0
        for sent_index, sent_word_count in enumerate(sent_word_counts):
            if sent_word_count == 0:
                continue
            next_index = flat_index + sent_word_count
            total_logprobs[sent_index, :sent_word_count] = flat_logprobs[flat_index:next_index]
            flat_index = next_index

        return total_logprobs


class CharProbRNNVarLenBatched(CharProbRNNVarLen):
    """
    Category-batched variant of CharProbRNNVarLen.

    It preserves the same scoring behavior while removing the Python loop over
    grammar categories by evaluating all (category, word) pairs in one LSTM.
    """

    def forward(self, chars, cat_embs, set_grammar=True):
        flat_inputs, flat_targets, sent_word_counts = self._flatten_words(chars)
        batch_size = len(chars)
        num_cats = cat_embs.shape[0]
        max_words = max(sent_word_counts, default=0)

        if not flat_inputs:
            return torch.empty(
                (batch_size, max_words, num_cats),
                device=cat_embs.device,
                dtype=self.char_embs.weight.dtype,
            )

        embeddings = [self.char_embs.forward(word) for word in flat_inputs]
        padded_char_embs = nn.utils.rnn.pad_sequence(embeddings)  # max_len, words, hidden
        num_words = len(flat_inputs)
        max_char_len = padded_char_embs.shape[0]
        word_lengths = [int(word.shape[0]) for word in flat_inputs]

        expanded_char_embs = padded_char_embs.unsqueeze(1).expand(
            max_char_len, num_cats, num_words, self.hidden_size
        )
        expanded_char_embs = expanded_char_embs.reshape(
            max_char_len, num_cats * num_words, self.hidden_size
        )
        packed_char_embs = nn.utils.rnn.pack_padded_sequence(
            expanded_char_embs,
            word_lengths * num_cats,
            enforce_sorted=False,
        )

        expanded_cat_embs = cat_embs.unsqueeze(1).expand(num_cats, num_words, -1)
        expanded_cat_embs = expanded_cat_embs.reshape(num_cats * num_words, -1)
        cat_tensor = self.cat_emb_expansion(expanded_cat_embs)  # cats*words, hidden*numlayers
        cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
        cat_tensor = cat_tensor.permute(2, 0, 1).contiguous()
        h0_tensor = torch.zeros_like(cat_tensor)

        all_hs, _ = self.rnn.forward(packed_char_embs, (h0_tensor, cat_tensor))
        all_hs, _ = nn.utils.rnn.pad_packed_sequence(all_hs, total_length=max_char_len)
        all_hs = all_hs.reshape(max_char_len, num_cats, num_words, self.hidden_size)
        Hses = all_hs.permute(1, 2, 0, 3).contiguous()  # cats, words, chars, hidden

        scores = self.top_fc.forward(Hses)  # cats, words, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)

        target_tensor = torch.zeros(
            (num_words, max_char_len),
            dtype=torch.long,
            device=logprobs.device,
        )
        target_mask = torch.zeros(
            (num_words, max_char_len),
            dtype=logprobs.dtype,
            device=logprobs.device,
        )
        for idx, targets in enumerate(flat_targets):
            target_len = targets.shape[0]
            target_tensor[idx, :target_len] = targets.to(logprobs.device)
            target_mask[idx, :target_len] = 1

        gathered = logprobs.permute(1, 0, 2, 3).gather(
            3,
            target_tensor[:, None, :, None].expand(num_words, num_cats, max_char_len, 1),
        ).squeeze(-1)
        flat_logprobs = (gathered * target_mask[:, None, :]).sum(-1)

        total_logprobs = flat_logprobs.new_zeros((batch_size, max_words, num_cats))
        flat_index = 0
        for sent_index, sent_word_count in enumerate(sent_word_counts):
            if sent_word_count == 0:
                continue
            next_index = flat_index + sent_word_count
            total_logprobs[sent_index, :sent_word_count] = flat_logprobs[flat_index:next_index]
            flat_index = next_index

        return total_logprobs


class CharProbRNNVarLenConditioned(CharProbRNNVarLen):
    """
    Character emission model for compound PCFG terminals.

    Unlike the simple-parser variants, category embeddings are sentence-specific:
    cat_embs has shape [batch, num_categories, state_dim].
    """

    def forward(self, chars, cat_embs, set_grammar=True):
        flat_inputs, flat_targets, sent_word_counts = self._flatten_words(chars)
        batch_size = len(chars)
        max_words = max(sent_word_counts, default=0)
        num_cats = cat_embs.shape[1]

        if not flat_inputs:
            return torch.empty(
                (batch_size, max_words, num_cats),
                device=cat_embs.device,
                dtype=self.char_embs.weight.dtype,
            )

        word_to_sent = []
        for sent_index, sent_word_count in enumerate(sent_word_counts):
            word_to_sent.extend([sent_index] * sent_word_count)
        word_to_sent = torch.tensor(word_to_sent, dtype=torch.long, device=cat_embs.device)

        embeddings = [self.char_embs.forward(word) for word in flat_inputs]
        padded_char_embs = nn.utils.rnn.pad_sequence(embeddings)  # max_len, words, hidden
        num_words = len(flat_inputs)
        max_char_len = padded_char_embs.shape[0]
        word_lengths = [int(word.shape[0]) for word in flat_inputs]

        expanded_char_embs = padded_char_embs.unsqueeze(2).expand(
            max_char_len, num_words, num_cats, self.hidden_size
        )
        expanded_char_embs = expanded_char_embs.reshape(
            max_char_len, num_words * num_cats, self.hidden_size
        )
        packed_char_embs = nn.utils.rnn.pack_padded_sequence(
            expanded_char_embs,
            [length for length in word_lengths for _ in range(num_cats)],
            enforce_sorted=False,
        )

        expanded_cat_embs = cat_embs.index_select(0, word_to_sent)  # words, cats, state_dim
        expanded_cat_embs = expanded_cat_embs.reshape(num_words * num_cats, -1)
        cat_tensor = self.cat_emb_expansion(expanded_cat_embs)  # words*cats, hidden*numlayers
        cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
        cat_tensor = cat_tensor.permute(2, 0, 1).contiguous()
        h0_tensor = torch.zeros_like(cat_tensor)

        all_hs, _ = self.rnn.forward(packed_char_embs, (h0_tensor, cat_tensor))
        all_hs, _ = nn.utils.rnn.pad_packed_sequence(all_hs, total_length=max_char_len)
        all_hs = all_hs.reshape(max_char_len, num_words, num_cats, self.hidden_size)
        Hses = all_hs.permute(1, 2, 0, 3).contiguous()  # words, cats, chars, hidden

        scores = self.top_fc.forward(Hses)  # words, cats, chars, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)

        target_tensor = torch.zeros(
            (num_words, max_char_len),
            dtype=torch.long,
            device=logprobs.device,
        )
        target_mask = torch.zeros(
            (num_words, max_char_len),
            dtype=logprobs.dtype,
            device=logprobs.device,
        )
        for idx, targets in enumerate(flat_targets):
            target_len = targets.shape[0]
            target_tensor[idx, :target_len] = targets.to(logprobs.device)
            target_mask[idx, :target_len] = 1

        gathered = logprobs.gather(
            3,
            target_tensor[:, None, :, None].expand(num_words, num_cats, max_char_len, 1),
        ).squeeze(-1)
        flat_logprobs = (gathered * target_mask[:, None, :]).sum(-1)  # words, cats

        total_logprobs = flat_logprobs.new_zeros((batch_size, max_words, num_cats))
        flat_index = 0
        for sent_index, sent_word_count in enumerate(sent_word_counts):
            if sent_word_count == 0:
                continue
            next_index = flat_index + sent_word_count
            total_logprobs[sent_index, :sent_word_count] = flat_logprobs[flat_index:next_index]
            flat_index = next_index

        return total_logprobs


class WordProbFCFixVocabCompound(nn.Module):
    def __init__(self, num_words, state_dim, dropout=0.0):
        super(WordProbFCFixVocabCompound, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, num_words))

    def forward(self, words, cat_embs, set_grammar=True):
        if set_grammar:
            dist = nn.functional.log_softmax(self.fc(cat_embs), 1).t() # vocab, cats
            self.dist = dist
        else:
            pass
        word_indices = words[:, 1:-1]

        logprobs = self.dist[word_indices, :] # sent, word, cats; get rid of bos and eos
        return logprobs


class ImageEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.no_imgnorm = opt.no_imgnorm
        hidden_dim = round(opt.img_dim/2+opt.sem_dim)
        self.mlp = nn.Sequential(
            torch.nn.Linear(opt.img_dim,  hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, opt.sem_dim)
        )
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, images):
        # why: assuming that the precomputed features are already l2-normalized
        features = self.mlp(images.float())
        if not self.no_imgnorm:
            features = l2norm(features)
        return features
    

class TextEncoder(torch.nn.Module):
    def __init__(self, opt, enc_emb=None):
        super(TextEncoder, self).__init__()
        self.NT = opt.nt_states
        self.sem_dim = opt.sem_dim
        self.syn_dim = opt.syn_dim
        self.enc_rnn = torch.nn.LSTM(opt.word_dim, opt.lstm_dim, 
            bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = torch.nn.Linear(
            opt.lstm_dim * 2, self.NT * self.sem_dim
        )
        self._initialize()
        self.enc_emb = enc_emb # avoid double initialization 

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def set_enc_emb(self, enc_emb):
        self.enc_emb = enc_emb

    def _forward_srnn(self, x_emb, lengths, spans=None):
        """ lstm over every span, a.k.a. segmental rnn 
        """
        b, N, dim = x_emb.size() # N longest sent len
        assert N == lengths.max()
        word_mask = torch.arange(
            0, N, device=x_emb.device
        ).unsqueeze(0).expand(b, N).long() 
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        feats = torch.zeros(
            b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=x_emb.device
        )
        beg_idx = 0 
        for k in range(1, N):
            inc = torch.arange(N - k, device=x_emb.device).view(N - k, 1)#.expand(N - k, k + 1)
            idx = torch.arange(k + 1, device=x_emb.device).view(1, k + 1).repeat(N - k, 1)
            idx = (idx + inc).view(-1)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(b, -1, dim) 

            feat = torch.gather(word_vect, 1, idx)
            feat = feat.view(b, N - k, k + 1, dim)
            feat = feat.view(-1, k + 1, dim) 
            feat = self.enc_out(self.enc_rnn(feat)[0])
            feat = feat.view(b, N - k, k + 1, self.NT, self.sem_dim)
            #sum LSTM output across each state for given string
            feat = l2norm(feat.sum(2))
            end_idx = beg_idx + N - k 
            feats[:, beg_idx : end_idx] = feat 
            beg_idx = end_idx
        return feats 
    
    def forward(self, x, lengths, spans):  
        word_emb = self.enc_emb(x)
        return self._forward_srnn(word_emb, lengths)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, img, txt):
        scores = self.sim(img, txt)
        diagonal = scores.diag().view(img.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(1)
        loss_img = loss_img.mean(0)
        return loss_txt + loss_img
