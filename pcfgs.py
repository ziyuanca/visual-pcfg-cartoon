import torch
from torch import nn
import torch.nn.functional as F
from nltk import tree as nltk_tree
import bidict
import numpy as np

from cky_parser_sgd import batch_CKY_parser, SMALL_NEGATIVE_NUMBER
from models import (
    CharProbRNN,
    CharProbRNNVarLen,
    CharProbRNNVarLenConditioned,
    WordProbFCFixVocabCompound,
    ResidualLayer,
)
from treenode import Node, nodes_to_tree

QUASI_INF = 10000000.0

def _extract_parse(ispan, length, inc=1):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    lprobs = []
    cover = ispan.nonzero()
    for i in range(cover.shape[0]):
        w, r, A = cover[i].tolist()
        a, b, _ = w, r, A
        w = w + inc
        r = r + w
        l = r - w
        spans.append((l, r, A))
        lprobs.append(ispan[a, b, A])
        if l != r:
            span = "({} {})".format(tree[l], tree[r])
            tree[r] = tree[l] = span
    return spans, tree[0], lprobs


def _extract_parses(matrix, lengths, inc=1):
    batch = matrix.shape[0]
    spans = []
    trees = []
    lprobs = []
    for b in range(batch):
        span, tree, lprob = _extract_parse(matrix[b], lengths[b], inc=inc)
        trees.append(tree)
        spans.append(span)
        lprobs.append(lprob)
    return spans, trees, lprobs


class SimpleCompPCFGCharNoDistinction(nn.Module):
    def __init__(self,
                 state_dim=64,
                 num_states=90,
                 num_chars=100,
                 device='cpu',
                 eval_device="cpu",
                 model_type='char',
                 num_words=100,
                 char_grams_lexicon=None,
                 all_words_char_features=None,
                 rnn_hidden_dim=320):
        super(SimpleCompPCFGCharNoDistinction, self).__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.model_type = model_type
        self.all_states = num_states
        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNNVarLen(
                num_chars, state_dim=self.state_dim, hidden_size=rnn_hidden_dim
            )
        elif self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocabCompound(num_words, state_dim)

        self.nont_emb = nn.Parameter(torch.randn(self.all_states, state_dim))

        self.rule_mlp = nn.Linear(state_dim, self.all_states ** 2)

        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        self.root_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, self.all_states))

        self.split_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, 2))

        self.device = device
        self.eval_device = eval_device
        self.pcfg_parser = batch_CKY_parser(nt=self.all_states, t=0, device=self.device)

    @staticmethod
    def _effective_parser_lengths(raw_lengths, batch_size, max_len, device):
        lengths = torch.tensor(raw_lengths, dtype=torch.long, device=device)
        if lengths.numel() != batch_size:
            raise ValueError(
                "lengths size mismatch: expected {}, got {}.".format(
                    batch_size, lengths.numel()
                )
            )
        # Parser operates over content tokens (BOS/EOS removed by emission models).
        lengths = (lengths - 2).clamp(min=1, max=max_len)
        return lengths

    def _marginal_grouped_by_length(
            self,
            emissions,
            effective_lengths,
            sent_indices=None,
            viterbi_flag=False,
            only_viterbi=False,
            return_span_posteriors=False,
            span_posteriors_log=False,
            exclude_length_one=False
    ):
        batch_size = emissions.shape[0]

        unique_lens = torch.unique(effective_lengths).tolist()
        if len(unique_lens) == 1:
            return self.pcfg_parser.marginal(
                emissions,
                viterbi_flag=viterbi_flag,
                only_viterbi=only_viterbi,
                sent_indices=sent_indices,
                return_span_posteriors=return_span_posteriors,
                span_posteriors_log=span_posteriors_log,
                exclude_length_one=exclude_length_one
            )

        logprob_full = (
            torch.empty(batch_size, device=emissions.device, dtype=emissions.dtype)
            if not only_viterbi else []
        )
        vtree_full = [None] * batch_size
        vprod_full = [None] * batch_size
        vbranch_full = [None] * batch_size
        span_full = None
        pad_value = -torch.inf if span_posteriors_log else 0.0
        max_len = int(effective_lengths.max().item())

        for span_len in sorted(unique_lens, reverse=True):
            select = (effective_lengths == span_len).nonzero(as_tuple=False).squeeze(1)
            local_ids = select.tolist()
            sub_emissions = emissions.index_select(0, select)[:, :span_len, :]
            sub_sent_indices = None if sent_indices is None else [sent_indices[i] for i in local_ids]

            outs = self.pcfg_parser.marginal(
                sub_emissions,
                viterbi_flag=viterbi_flag,
                only_viterbi=only_viterbi,
                sent_indices=sub_sent_indices,
                return_span_posteriors=return_span_posteriors,
                span_posteriors_log=span_posteriors_log,
                exclude_length_one=exclude_length_one
            )

            if return_span_posteriors:
                logprob_list, vtree_list, vprod_list, vbranch_list, span_post = outs
            else:
                logprob_list, vtree_list, vprod_list, vbranch_list = outs

            if not only_viterbi:
                logprob_full.index_copy_(0, select, logprob_list)
            for i, global_id in enumerate(local_ids):
                vtree_full[global_id] = vtree_list[i]
                vprod_full[global_id] = vprod_list[i]
                vbranch_full[global_id] = vbranch_list[i]

            if return_span_posteriors:
                if span_full is None:
                    if span_post.dim() != 4:
                        raise ValueError(
                            "Expected labeled span posteriors with rank 4, got {}.".format(
                                span_post.dim()
                            )
                        )
                    span_full = torch.full(
                        (batch_size, max_len, max_len, span_post.shape[-1]),
                        pad_value,
                        device=span_post.device,
                        dtype=span_post.dtype
                    )

                span_full[select, :span_len, :span_len, :] = span_post

        if return_span_posteriors:
            return logprob_full, vtree_full, vprod_full, vbranch_full, span_full
        return logprob_full, vtree_full, vprod_full, vbranch_full

    def forward(
        self,
        x,
        lengths,
        eval=False,
        argmax=False,
        use_mean=False,
        indices=None,
        set_grammar=True,
        return_ll=True,
        return_span_posteriors=False,
        **kwargs
    ):
        if set_grammar:
            self.emission = None

            nt_emb = self.nont_emb

            root_scores = F.log_softmax(self.root_mlp(self.root_emb).squeeze(), 0)
            full_p0 = root_scores

            # rule_score = F.log_softmax(self.rule_mlp([nt_emb, nt_emb, nt_emb]).squeeze().reshape([self.all_states, self.all_states**2]), dim=1)
            rule_score = F.log_softmax(self.rule_mlp(nt_emb), 1)  # nt x t**2

            full_G = rule_score
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)
            full_G = full_G + split_scores[:, 0][..., None]

            self.pcfg_parser.set_models(full_p0, full_G, self.emission, pcfg_split=split_scores)

        x = self.emit_prob_model(x, self.nont_emb, set_grammar=set_grammar)
        effective_lengths = self._effective_parser_lengths(
            lengths,
            x.shape[0],
            x.shape[1],
            x.device
        )

        if argmax:
            if eval and self.device != self.eval_device:
                self.pcfg_parser.device = self.eval_device

            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self._marginal_grouped_by_length(
                        x,
                        effective_lengths,
                        sent_indices=indices,
                        viterbi_flag=True,
                        only_viterbi=not return_ll
                    )
            if eval and self.device != self.eval_device:
                self.pcfg_parser.device = self.device

            return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
        else:
            if return_span_posteriors:
                logprob_list, _, _, _, span_posteriors = self._marginal_grouped_by_length(
                    x,
                    effective_lengths,
                    sent_indices=indices,
                    return_span_posteriors=True,
                    span_posteriors_log=True,
                    exclude_length_one=True
                )
                logprob_list = logprob_list * (-1)
                return logprob_list, span_posteriors
            logprob_list, _, _, _ = self._marginal_grouped_by_length(
                x,
                effective_lengths,
                sent_indices=indices
            )
            logprob_list = logprob_list * (-1)
            return logprob_list


class CompoundPCFG(nn.Module):
    def __init__(
        self,
        state_dim=256,
        nt_states=30,
        t_states=60,
        z_dim=64,
        h_dim=512,
        w_dim=512,
        device="cpu",
        eval_device="cpu",
        model_type="word",
        num_words=100,
        num_chars=100,
        rnn_hidden_dim=320,
    ):
        super(CompoundPCFG, self).__init__()
        if model_type not in ("word", "char"):
            raise ValueError("CompoundPCFG currently supports model_type='word' or 'char'.")
        try:
            from torch_struct import SentCFG
        except Exception as exc:
            raise ImportError(
                "parser_type=compound requires torch_struct. Please install torch-struct."
            ) from exc

        # Older torch-struct builds may not define this metadata directly on the
        # class, which triggers a harmless warning from torch.distributions.
        if "arg_constraints" not in SentCFG.__dict__:
            SentCFG.arg_constraints = {}

        self._SentCFG = SentCFG
        self.model_type = model_type
        self.posterior_model_type = "word"
        self.device = device
        self.eval_device = eval_device

        self.NT = nt_states
        self.T = t_states
        self.all_states = nt_states + t_states
        self.matching_nt_states = self.NT
        self.uses_flat_span_marginals = True

        self.z_dim = z_dim
        self.s_dim = state_dim
        self.term_state_dim = self.s_dim + self.z_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))

        self.rule_mlp = nn.Linear(self.s_dim + self.z_dim, self.all_states ** 2)
        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim + self.z_dim, self.s_dim),
            ResidualLayer(self.s_dim, self.s_dim),
            ResidualLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )
        if self.model_type == "word":
            self.term_mlp = nn.Sequential(
                nn.Linear(self.s_dim + self.z_dim, self.s_dim),
                ResidualLayer(self.s_dim, self.s_dim),
                ResidualLayer(self.s_dim, self.s_dim),
                nn.Linear(self.s_dim, num_words),
            )
            self.term_char_model = None
        elif self.model_type == "char":
            self.term_mlp = None
            self.term_char_model = CharProbRNNVarLenConditioned(
                num_chars,
                state_dim=self.term_state_dim,
                hidden_size=rnn_hidden_dim,
            )
        if self.z_dim > 0:
            self.enc_emb = nn.Embedding(num_words, w_dim)
            self.enc_rnn = nn.LSTM(
                w_dim, h_dim, bidirectional=True, num_layers=1, batch_first=True
            )
            self.enc_out = nn.Linear(h_dim * 2, self.z_dim * 2)
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def _kl(mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def _enc(self, x):
        x_embbed = self.enc_emb(x)
        h, _ = self.enc_rnn(x_embbed)
        out = self.enc_out(h.max(1)[0])
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    @staticmethod
    def _trim_bos_eos(x):
        if x.size(1) < 3:
            raise ValueError("Expected BOS/EOS in sequence for compound parser.")
        return x[:, 1:-1]

    def _forward_params(self, x, chars=None, use_mean=False):
        b, n = x.shape[:2]
        if self.z_dim > 0:
            mean, lvar = self._enc(x)
            kl = self._kl(mean, lvar).sum(1)
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
        else:
            z = torch.zeros(b, 1, device=x.device)
            kl = torch.zeros(b, device=x.device)

        root_emb = self.root_emb.expand(b, self.s_dim)
        if self.z_dim > 0:
            root_emb = torch.cat([root_emb, z], -1)
        roots_ll = F.log_softmax(self.root_mlp(root_emb), -1)

        term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(b, n, self.T, self.s_dim)
        if self.z_dim > 0:
            z_expand = z.unsqueeze(1).unsqueeze(2).expand(b, n, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
        if self.model_type == "word":
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            terms_ll = torch.gather(term_prob, 3, indices).squeeze(3)
        else:
            if chars is None:
                raise ValueError("CompoundPCFG with model_type='char' requires chars input.")
            if len(chars) != b:
                raise ValueError(
                    "chars batch size mismatch: expected {}, got {}.".format(b, len(chars))
                )
            term_state_emb = term_emb[:, 0]  # batch, T, state_dim(+z_dim), shared across words
            terms_ll = self.term_char_model(chars, term_state_emb)
            if terms_ll.shape[1] != n:
                raise ValueError(
                    "char emissions length mismatch: expected {}, got {}.".format(
                        n, terms_ll.shape[1]
                    )
                )

        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(b, self.NT, self.s_dim)
        if self.z_dim > 0:
            z_expand = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
            nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
        rules_ll = F.log_softmax(self.rule_mlp(nonterm_emb), -1)
        rules_ll = rules_ll.view(b, self.NT, self.all_states, self.all_states)

        return (terms_ll, rules_ll, roots_ll), kl

    def forward(
        self,
        x,
        lengths,
        eval=False,
        argmax=False,
        use_mean=False,
        indices=None,
        set_grammar=True,
        return_ll=True,
        return_span_posteriors=False,
        **kwargs
    ):
        del eval, indices, set_grammar, return_ll
        chars = kwargs.get("chars", None)
        # The existing pipeline uses BOS/EOS; compound parser operates on content words only.
        x_trim = self._trim_bos_eos(x)
        lengths = torch.tensor(lengths, dtype=torch.long, device=x_trim.device)
        lengths = (lengths - 2).clamp(min=1, max=x_trim.size(1))

        params, kl = self._forward_params(x_trim, chars=chars, use_mean=use_mean)
        dist = self._SentCFG(params, lengths=lengths)
        if x_trim.size(1) <= 1:
            # For single-word sentences, rule potentials are structurally unused.
            # `inside_im` may fail in some torch-struct builds when requesting
            # marginals through autograd in this case.
            ll = dist.partition
            span_margs = torch.empty(
                (x_trim.size(0), 0, self.NT),
                device=x_trim.device,
                dtype=ll.dtype
            )
        else:
            ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        kl = kl.clamp(max=20) # avoid kl explosion
        logprob_list = nll + kl

        if argmax:
            with torch.no_grad():
                if x_trim.size(1) <= 1:
                    root_ids = torch.argmax(params[2], dim=-1).tolist()
                    tree_list = [nltk_tree.Tree.fromstring(f"({rid} 0)") for rid in root_ids]
                else:
                    argmax_parts = dist.argmax
                    term_parts = argmax_parts[0]
                    the_spans = argmax_parts[-1]
                    span_lists, _, _ = _extract_parses(the_spans, lengths.tolist(), inc=0)
                    root_ids = torch.argmax(params[2], dim=-1).tolist()
                    tree_list = []
                    for b, sent_spans in enumerate(span_lists):
                        sent_len = lengths[b].item()
                        if sent_len <= 1:
                            tree_list.append(nltk_tree.Tree.fromstring(f"({root_ids[b]} 0)"))
                            continue
                        term_ids = torch.argmax(term_parts[b, :sent_len], dim=-1).tolist()
                        nodes = [
                            Node(cat, index, index + 1, D=-1, K=self.NT)
                            for index, cat in enumerate(term_ids)
                        ]
                        nodes.extend(
                            Node(cat, left, right + 1, D=-1, K=self.NT)
                            for (left, right, cat) in sent_spans
                            if left != right
                        )
                        if not nodes:
                            tree_list.append(nltk_tree.Tree.fromstring(f"({root_ids[b]} 0)"))
                            continue
                        this_tree, _, _ = nodes_to_tree(nodes, x_trim[b, : sent_len])
                        tree_list.append(this_tree)
            return logprob_list, tree_list, [None] * x.shape[0], [None] * x.shape[0]

        if return_span_posteriors:
            if span_margs.size(-1) > self.NT:
                raise ValueError(
                    "Expected span marginals over at most {} nonterminal states, got {}.".format(
                        self.NT,
                        span_margs.size(-1)
                    )
                )
            return logprob_list, span_margs
        return logprob_list
