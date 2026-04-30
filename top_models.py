import torch.nn as nn
import torch.distributions
import torch
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from treenode import convert_binary_matrix_to_strtree
from models import ImageEncoder, TextEncoder, ContrastiveLoss, l2norm

DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class TopModel(nn.Module):
    def __init__(
        self,
        inducer,
        writer: SummaryWriter = None,
        config=None,
        vocab_size=None,
        image_dim=None,
    ):
        super(TopModel, self).__init__()
        self.inducer = inducer
        self.writer = writer
        self.joint_training = False
        self.vse_mt_alpha = 0.0
        self.vse_lm_alpha = 1.0
        self.img_enc = None
        self.txt_enc = None
        self.loss_criterion = None
        self.vectorized_span_matching = False
        self.span_matching_chunk_size = 0

        if config is not None and vocab_size is not None:
            self.joint_training = config.getboolean("joint_training", fallback=False)
            self.vse_mt_alpha = config.getfloat("vse_mt_alpha", fallback=0.0)
            self.vse_lm_alpha = config.getfloat("vse_lm_alpha", fallback=1.0)
            self.vectorized_span_matching = config.getboolean(
                "vectorized_span_matching",
                fallback=False,
            )
            self.span_matching_chunk_size = config.getint(
                "span_matching_chunk_size",
                fallback=0,
            )
            if self.joint_training:
                if image_dim is None:
                    raise ValueError("image_dim is required when joint_training=true.")
                matching_nt_states = getattr(self.inducer, "matching_nt_states", self.inducer.all_states)
                enc_opt = SimpleNamespace(
                    nt_states=matching_nt_states,
                    sem_dim=config.getint("sem_dim", fallback=256),
                    syn_dim=config.getint("syn_dim", fallback=256),
                    word_dim=config.getint("word_dim", fallback=256),
                    lstm_dim=config.getint("lstm_dim", fallback=256),
                    img_dim=image_dim,
                    no_imgnorm=config.getboolean("no_imgnorm", fallback=False),
                )
                word_emb = torch.nn.Embedding(vocab_size, enc_opt.word_dim)
                torch.nn.init.xavier_uniform_(word_emb.weight)
                self.img_enc = ImageEncoder(enc_opt)
                self.txt_enc = TextEncoder(enc_opt, word_emb)
                self.loss_criterion = ContrastiveLoss(margin=config.getfloat("margin", fallback=0.2))

    @staticmethod
    def _flatten_labeled_span_log_post(span_log_post):
        # span_log_post: [batch, start, end_inclusive, state]
        batch_size, sent_len, _, num_states = span_log_post.shape
        flat = []
        for height in range(1, sent_len):  # length >= 2
            starts = torch.arange(sent_len - height, device=span_log_post.device)
            ends = starts + height
            flat.append(span_log_post[:, starts, ends, :])
        if not flat:
            return torch.empty(
                (batch_size, 0, num_states),
                device=span_log_post.device,
                dtype=span_log_post.dtype
        )
        return torch.cat(flat, dim=1)

    def _run_inducer(self, word_inp, chars_var_inp, **kwargs):
        if self.inducer.model_type == "char":
            if getattr(self.inducer, "posterior_model_type", None) == "word":
                return self.inducer.forward(word_inp, chars=chars_var_inp, **kwargs)
            return self.inducer.forward(chars_var_inp, words=word_inp, **kwargs)
        assert self.inducer.model_type == 'word'
        return self.inducer.forward(word_inp, **kwargs)


    def forward(self, word_inp, chars_var_inp, *, lengths, distance_penalty_weight=0., images=None):
        if self.joint_training:
            if images is None:
                raise ValueError("joint_training=true requires image batches in TopModel.forward().")
            if self.img_enc is None or self.txt_enc is None or self.loss_criterion is None:
                raise ValueError(
                    "joint_training=true but matching modules were not initialized."
                )
            use_matching = True
        else:
            if images is not None:
                raise ValueError(
                    "Received image batches with joint_training=false in TopModel.forward()."
                )
            use_matching = False
        if use_matching:
            logprob_list, span_log_post = self._run_inducer(
                word_inp,
                chars_var_inp,
                lengths=lengths,
                return_span_posteriors=True,
            )
        else:
            logprob_list = self._run_inducer(word_inp, chars_var_inp, lengths=lengths)

        structure_loss = torch.sum(logprob_list, dim=0)
        matching_loss = torch.zeros_like(structure_loss)

        if use_matching:
            lengths = torch.tensor(lengths, dtype=torch.long, device=word_inp.device)

            img_emb = self.img_enc(images)
            trimmed_word_inp = word_inp[:, 1:-1]
            trimmed_lengths = (lengths - 2).clamp(min=1)
            cap_span_features = self.txt_enc(trimmed_word_inp, trimmed_lengths, None)
            if getattr(self.inducer, "uses_flat_span_marginals", False):
                span_margs = span_log_post
                if self.vectorized_span_matching:
                    matching_loss = self._flat_span_matching_loss_vectorized(
                        img_emb,
                        cap_span_features,
                        span_margs,
                    )
                else:
                    matching_loss = self._flat_span_matching_loss_loop(
                        img_emb,
                        cap_span_features,
                        span_margs,
                    )
            else:
                span_log_flat = self._flatten_labeled_span_log_post(span_log_post)
                if self.vectorized_span_matching:
                    matching_loss = self._log_span_matching_loss_vectorized(
                        img_emb,
                        cap_span_features,
                        span_log_flat,
                    )
                else:
                    matching_loss = self._log_span_matching_loss_loop(
                        img_emb,
                        cap_span_features,
                        span_log_flat,
                    )

        bsize = word_inp.shape[0]
        total_loss = (
            self.vse_lm_alpha * structure_loss + self.vse_mt_alpha * matching_loss
        ) / bsize
        return total_loss

    def _flat_span_matching_loss_loop(self, img_emb, cap_span_features, span_margs):
        nstep = min(cap_span_features.shape[1], span_margs.shape[1])
        if nstep <= 0:
            return img_emb.new_zeros(())

        matching_loss_matrix = torch.zeros(
            (img_emb.shape[0], nstep),
            device=img_emb.device,
            dtype=img_emb.dtype,
        )
        for k in range(nstep):
            cap_emb = cap_span_features[:, k]
            span_state_mass = span_margs[:, k]
            span_mass = span_state_mass.sum(-1)
            cap_marg = torch.where(
                span_mass[:, None] > 0,
                span_state_mass / span_mass.clamp(min=1e-8)[:, None],
                torch.zeros_like(span_state_mass),
            ).unsqueeze(-2)
            cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)
            cap_emb = l2norm(cap_emb)
            matching_loss_matrix[:, k] = (
                self.loss_criterion(img_emb, cap_emb) * span_mass
            )
        return matching_loss_matrix.sum()

    def _flat_span_matching_loss_vectorized(self, img_emb, cap_span_features, span_margs):
        nstep = min(cap_span_features.shape[1], span_margs.shape[1])
        if nstep <= 0:
            return img_emb.new_zeros(())

        matching_loss = img_emb.new_zeros(())
        for start, end in self._span_matching_chunks(nstep):
            cap_chunk = cap_span_features[:, start:end]
            span_state_mass = span_margs[:, start:end]
            span_mass = span_state_mass.sum(-1)
            cap_weights = torch.where(
                span_mass[..., None] > 0,
                span_state_mass / span_mass.clamp(min=1e-8)[..., None],
                torch.zeros_like(span_state_mass),
            )
            cap_emb = torch.einsum("bsn,bsnd->bsd", cap_weights, cap_chunk)
            cap_emb = l2norm(cap_emb)
            matching_loss = matching_loss + (
                self._contrastive_loss_by_span_vectorized(img_emb, cap_emb) * span_mass
            ).sum()
        return matching_loss

    def _log_span_matching_loss_loop(self, img_emb, cap_span_features, span_log_flat):
        nstep = min(cap_span_features.shape[1], span_log_flat.shape[1])
        if nstep <= 0:
            return img_emb.new_zeros(())

        matching_loss_matrix = torch.zeros(
            (img_emb.shape[0], nstep),
            device=img_emb.device,
            dtype=img_emb.dtype,
        )
        for k in range(nstep):
            span_log_k = span_log_flat[:, k]
            finite_mask = torch.isfinite(span_log_k)
            valid_span = finite_mask.any(dim=-1)
            safe_span_log_k = torch.where(
                finite_mask,
                span_log_k,
                torch.full_like(span_log_k, -1e8)
            )
            span_log_z = torch.logsumexp(safe_span_log_k, dim=-1)
            span_prob = torch.where(
                valid_span,
                span_log_z.exp(),
                torch.zeros_like(span_log_z)
            )
            cap_marg = torch.softmax(safe_span_log_k, dim=-1).unsqueeze(-2)
            cap_emb = torch.matmul(cap_marg, cap_span_features[:, k]).squeeze(-2)
            cap_emb = l2norm(cap_emb)
            matching_loss_matrix[:, k] = self.loss_criterion(img_emb, cap_emb) * span_prob
        return matching_loss_matrix.sum()

    def _log_span_matching_loss_vectorized(self, img_emb, cap_span_features, span_log_flat):
        nstep = min(cap_span_features.shape[1], span_log_flat.shape[1])
        if nstep <= 0:
            return img_emb.new_zeros(())

        matching_loss = img_emb.new_zeros(())
        for start, end in self._span_matching_chunks(nstep):
            cap_chunk = cap_span_features[:, start:end]
            span_log_chunk = span_log_flat[:, start:end]
            finite_mask = torch.isfinite(span_log_chunk)
            valid_span = finite_mask.any(dim=-1)
            safe_span_log = torch.where(
                finite_mask,
                span_log_chunk,
                torch.full_like(span_log_chunk, -1e8),
            )
            span_log_z = torch.logsumexp(safe_span_log, dim=-1)
            span_prob = torch.where(
                valid_span,
                span_log_z.exp(),
                torch.zeros_like(span_log_z),
            )
            cap_weights = torch.softmax(safe_span_log, dim=-1)
            cap_emb = torch.einsum("bsn,bsnd->bsd", cap_weights, cap_chunk)
            cap_emb = l2norm(cap_emb)
            matching_loss = matching_loss + (
                self._contrastive_loss_by_span_vectorized(img_emb, cap_emb) * span_prob
            ).sum()
        return matching_loss

    def _contrastive_loss_by_span_vectorized(self, img_emb, cap_emb):
        batch_size, span_count, _ = cap_emb.shape
        if batch_size == 0 or span_count == 0:
            return img_emb.new_zeros((batch_size, span_count))

        scores = torch.einsum("id,jsd->isj", img_emb, cap_emb)
        batch_ids = torch.arange(batch_size, device=scores.device)
        diagonal = scores[batch_ids, :, batch_ids]
        loss_txt = (
            self.loss_criterion.margin + scores - diagonal[:, :, None]
        ).clamp(min=self.loss_criterion.min_val)
        loss_img = (
            self.loss_criterion.margin + scores - diagonal.transpose(0, 1)[None, :, :]
        ).clamp(min=self.loss_criterion.min_val)

        identity = torch.eye(batch_size, dtype=torch.bool, device=scores.device)
        identity = identity[:, None, :].expand(batch_size, span_count, batch_size)
        loss_txt = loss_txt.masked_fill(identity, 0)
        loss_img = loss_img.masked_fill(identity, 0)
        return loss_txt.mean(2) + loss_img.mean(0).transpose(0, 1)

    def _span_matching_chunks(self, nstep):
        chunk_size = int(self.span_matching_chunk_size or 0)
        if chunk_size <= 0 or chunk_size >= nstep:
            yield 0, nstep
            return
        for start in range(0, nstep, chunk_size):
            yield start, min(start + chunk_size, nstep)


    def parse(self, word_inp, chars_var_inp, indices, *, lengths, eval=False, set_grammar=True):
        inference_use_mean = getattr(self.inducer, "z_dim", 0) > 0 # turn on use_mean if using latent variables
        printDebug("word input: {}".format(word_inp))
        structure_loss, vtree_list, _, _ = self._run_inducer(
            word_inp,
            chars_var_inp,
            eval=eval,
            argmax=True,
            use_mean=inference_use_mean,
            indices=indices,
            set_grammar=set_grammar,
            lengths=lengths,
        )
        score = structure_loss.sum().item()
        # Keep simple parser behavior unchanged; make compound parser parse-time
        # score comparable (higher is better) for existing logging/early stop flow.
        if getattr(self.inducer, "uses_flat_span_marginals", False):
            score = -score
        return score, vtree_list


    def likelihood(self, word_inp, chars_var_inp, indices, *, lengths, set_grammar=True):
        inference_use_mean = getattr(self.inducer, "z_dim", 0) > 0
        structure_loss = self._run_inducer(
            word_inp,
            chars_var_inp,
            argmax=False,
            use_mean=inference_use_mean,
            indices=indices,
            set_grammar=set_grammar,
            lengths=lengths,
        )
        return structure_loss.sum().item()
