from selectors import EpollSelector
from turtle import st
from numpy import indices
from sympy import Trace, false, true
import torch
import torch.profiler
import torch.nn as nn

from x_transformers import Decoder

from .transformer_utils import MultiEmbedding, RVQMultiEmbedding
from .sub_decoder_utils import *
from .sampling_utils import sample, sample_with_prob, sample_with_prob_fast, top_p_sampling, typical_sampling, eta_sampling

from data_representation.vocab_utils import LangTokenVocab

class SingleProjection(nn.Module):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    This sub-decoder is used for REMI based models
    '''
    super().__init__()
    vocab_size = vocab.get_vocab_size()
    self.proj = nn.Linear(dim, vocab_size)
    
  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=1):
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']
    # ---- Generate(Inference) ---- #
    if target is None:
      logits = self.proj(hidden_vec[:, -1:])
      sampled_token = sample(logits, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      return logits, sampled_token
    # ---- Training ---- #
    logits = self.proj(hidden_vec)
    return logits

class SubDecoderClass(nn.Module):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    super().__init__()
    '''
    This is the base class for all sub-decoders
    '''
    self.prediction_order = prediction_order
    self.vocab = vocab
    self.vocab_size = vocab.get_vocab_size()
    # make layers
    self._make_emb_layer(vocab, dim)
    self._make_projection_layer(vocab, dim)
    self._make_nonlinear_layer()

  @property
  def device(self):
    return next(self.parameters()).device

  def _make_emb_layer(self, vocab, dim):
    self.emb_layer = MultiEmbedding(
      vocab=vocab,
      dim_model=dim
    )

  # def _make_projection_layer(self, vocab, dim):
  #   vocab_sizes = vocab.get_vocab_size()
  #   self.hidden2logit = nn.ModuleDict({
  #     f"layer_{key}": nn.Linear(dim, size) for key, size in vocab_sizes.items()
  #     })

  def _make_nonlinear_layer(self):
    pass
  def _make_projection_layer(self, vocab, dim):
      vocab_sizes = vocab.get_vocab_size()
      self.vocab_sizes = vocab_sizes
      self.max_vocab_size = max(vocab_sizes.values())
      self.projection_keys = list(vocab_sizes.keys())  # For index order

      # ✅ 保留原来的 Linear 层（这样 state_dict 可以匹配）
      self.hidden2logit = nn.ModuleDict({
          f"layer_{key}": nn.Linear(dim, size) for key, size in vocab_sizes.items()
      })

      # # ✅ 构建用于 block 并行的权重
      # weight_list = []
      # bias_list = []

      # for key in self.projection_keys:
      #     layer = self.hidden2logit[f"layer_{key}"]
      #     w = layer.weight
      #     b = layer.bias

      #     # pad to max_vocab_size
      #     w_padded = F.pad(w, (0, 0, 0, self.max_vocab_size - w.shape[0]))
      #     b_padded = F.pad(b, (0, self.max_vocab_size - b.shape[0]))

      #     weight_list.append(w_padded.unsqueeze(0))  # (1, Vmax, D)
      #     bias_list.append(b_padded.unsqueeze(0))    # (1, Vmax)

      # self.register_buffer("proj_weight", torch.cat(weight_list, dim=0))  # (F, Vmax, D)
      # self.register_buffer("proj_bias", torch.cat(bias_list, dim=0))      # (F, Vmax)  
class FeedForward(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    FeedForward sub-decoder is used for compound token like CP or NB.
    We followed the original sub-decoder proposed in the paper "Compound Word Transformer",
    however the embedding size for each sub-token or musical feature is the same in our implementation.
    The reason for that is we didn't find any significant difference in the performance of the model

    There are two types of decoding style for the FeedForward sub-decoder:
    1. Partial-sequential prediction: Predict type token first and then predict all the sub-tokens in parallel (origianl CP)
    2. Fully-sequential prediction: Predict all the sub-tokens sequentially
    '''
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def _make_projection_layer(self, vocab, dim):
    vocab_sizes = vocab.get_vocab_size()
    self.hidden2logit = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim, size) for key, size in vocab_sizes.items()
      })
    self.catvec2hidden = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim+dim, dim) for key, _ in vocab_sizes.items()
      })

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']

    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      for feature in self.prediction_order:
        if isinstance(feature, str):
          logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
          logits_dict[feature] = logit
          sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
          sampled_token_dict[feature] = sampled_token
          feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token) # B x T x emb_size
          catvec = torch.cat([hidden_vec, feature_emb.unsqueeze(0)], dim=-1)
          hidden_vec = self.catvec2hidden[f"layer_{feature}"](catvec)
        else:
          assert feature == self.prediction_order[-1], "Parallel prediction should be the last feature"
          for par_feature in feature:
            logit = self.hidden2logit[f"layer_{par_feature}"](hidden_vec)
            logits_dict[par_feature] = logit
            sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
            sampled_token_dict[par_feature] = sampled_token
      return logits_dict, sampled_token_dict

    # ---- Training ---- #
    for feature in self.prediction_order:
      if isinstance(feature, str):
        logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
        logits_dict[feature] = logit
        feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., self.vocab.feature_list.index(feature)]) # B x T x emb_size
        catvec = torch.cat([hidden_vec, feature_emb], dim=-1)
        hidden_vec = self.catvec2hidden[f"layer_{feature}"](catvec)
      else:
        assert feature == self.prediction_order[-1], "Parallel prediction should be the last feature"
        for par_feature in feature:
          logit = self.hidden2logit[f"layer_{par_feature}"](hidden_vec)
          logits_dict[par_feature] = logit
    return logits_dict

class Parallel(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    Parallel sub-decoder is used for parallel prediction of multiple sub-tokens or musical features
    This method is proposed in the paper "Multitrack Music Transformer"
    '''
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']

    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      for feature in self.prediction_order:
        logit = self.hidden2logit[f"layer_{feature}"](hidden_vec) # B x T x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    for feature in self.prediction_order:
      logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
      logits_dict[feature] = logit
    return logits_dict

class RNN(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    RNN sub-decoder is used for sequential prediction of multiple sub-tokens or musical features
    This method is similar to the method proposed in "PianoTree VAE"
    '''
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}

    self.pos_enc = nn.Embedding(len(prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)

    self.decoding_rnn = nn.GRU(
                      input_size=dim,
                      hidden_size=dim,
                      num_layers=sub_decoder_depth,
                      dropout=dropout,
                      batch_first=True)

  def _apply_pos_enc(self, tgt, apply_type='last'):
    if apply_type == 'all':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      tgt_pos = tgt + self.pos_enc(pos.long())
    elif apply_type == 'last':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      pos_emb = self.pos_enc(pos.long())
      # zero out the pos_emb except for the last token
      pos_emb[:, :-1, :] = 0
      tgt_pos = tgt + pos_emb
    return tgt_pos

  def _prepare_token_embedding_for_teacher_forcing(self, input_seq, target):
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq = torch.cat([input_seq, feature_emb_reshape], dim=1) 
    return input_seq

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x num_sub_tokens-1
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], -1)).unsqueeze(1) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape # (B*T) x 1 x d_model
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      h_0 = input_seq[:, 0, :].unsqueeze(0) # 1 x (B*T) x d_model
      input_seq = self._apply_pos_enc(input_seq, apply_type='all') # (B*T) x 1 x d_model
      for idx, feature in enumerate(self.prediction_order):
        input_seq, _ = self.decoding_rnn(input_seq, h_0) # input_seq: (B*T) x (idx+1) x hidden_size, h_n: num_layers x (B*T) x hidden_size
        logit = self.hidden2logit[f"layer_{feature}"](input_seq[:, -1, :]) # (B*T) x vocab_size
        logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token) # B x T x emb_size
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        input_seq = torch.cat([input_seq, feature_emb_reshape], dim=1) # (B*T) x (idx+2) x d_model
        input_seq = self._apply_pos_enc(input_seq, apply_type='last') # (B*T) x (idx+2) x d_model
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    input_seq = self._prepare_token_embedding_for_teacher_forcing(input_seq, target) # (B*T) x len(prediction_order) x d_model
    # initial hidden state has no positional encoding
    h0 = input_seq[:, 0, :].unsqueeze(0) # 1 x (B*T) x d_model 
    h0 = h0.contiguous()
    # apply positional encoding
    input_seq = self._apply_pos_enc(input_seq, apply_type='all') # (B*T) x len(prediction_order) x d_model
    # get output using rnn
    output, _ = self.decoding_rnn(input_seq, h0) # (B*T) x len(prediction_order) x d_model
    output = output.reshape((hidden_vec.shape[0], hidden_vec.shape[1], len(self.prediction_order), -1)) # B x T x len(prediction_order) x d_model
    for idx, feature in enumerate(self.prediction_order):
      logit = self.hidden2logit[f"layer_{feature}"](output[:, :, idx, :]) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class SelfAttention(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    This sub-decoder is used for sequential prediction of multiple sub-tokens or musical features
    This method is similar to the method proposed in "UniAudio", but different in making the sequence of sub-tokens.
    The UniAudio adds the output of the main decoder or hidden vec directly to embedding of the sub-token,
    while our method puts the hidden vec in the input sequence so that the attention mechanism can learn the relationship between the hidden vec and the sub-token
    '''
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(1 + len(prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)
    
    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    
    window_size = 1 # number of previous output of the main decoder to be used in the sub-decoder
    causal_mask = generate_causality_mask_on_window(size=window_size + len(prediction_order), window_size=window_size)
    self.register_buffer('causal_mask', causal_mask)

    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = sub_decoder_depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True)
    # add final dropout
    print('Applying Xavier Uniform Init to x-transformer following torch.Transformer')
    self._apply_xavier_init()
    print('Adding dropout after feedforward layer in x-transformer')
    self._add_dropout_after_ff(dropout)
    print('Adding dropout after attention layer in x-transformer')
    self._add_dropout_after_attn(dropout)

  def _add_dropout_after_attn(self, dropout):
    for layer in self.transformer_decoder.layers:
      if 'Attention' in str(type(layer[1])): 
        if isinstance(layer[1].to_out, nn.Sequential): # if GLU
          layer[1].to_out.append(nn.Dropout(dropout))
        elif isinstance(layer[1].to_out, nn.Linear): # if simple linear
          layer[1].to_out = nn.Sequential(layer[1].to_out, nn.Dropout(dropout))
        else:
          raise ValueError('to_out should be either nn.Sequential or nn.Linear')

  def _add_dropout_after_ff(self, dropout):
    for layer in self.transformer_decoder.layers:
      if 'FeedForward' in str(type(layer[1])):
        layer[1].ff.append(nn.Dropout(dropout))

  def _apply_xavier_init(self):
    for name, param in self.transformer_decoder.named_parameters():
      if 'to_q' in name or 'to_k' in name or 'to_v' in name:
          torch.nn.init.xavier_uniform_(param, gain=0.5**0.5)

  def _apply_pos_enc(self, tgt, apply_type='last'):
    if apply_type == 'all':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      tgt_pos = tgt + self.pos_enc(pos.long())
    elif apply_type == 'last':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      pos_emb = self.pos_enc(pos.long()) # (B*T) x (window_size + BOS + num_sub_tokens-1) x dim
      # zero out the pos_emb except for the last token
      pos_emb[:, :-1, :] = 0
      tgt_pos = tgt + pos_emb
    return tgt_pos

  def _prepare_input_seq_list(self, hidden_vec_reshape, target=None):
    input_seq_list = []
    input_seq_list.append(hidden_vec_reshape)
    BOS_emb = self.sub_decoder_BOS_emb.unsqueeze(0).repeat(hidden_vec_reshape.shape[0], 1, 1) # (B*T) x 1 x d_model
    if target is None:
      input_seq_list.append(BOS_emb[-1:, :, :])
    else: # training
      input_seq_list.append(BOS_emb)
    return input_seq_list

  def _prepare_token_embedding_for_teacher_forcing(self, input_seq_list, target):
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS + num_sub_tokens-1) x d_model
    return memory_tensor

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x num_sub_tokens
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq_list = self._prepare_input_seq_list(hidden_vec_reshape, target)
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      input_seq_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS) x d_model
      pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      for idx, feature in enumerate(self.prediction_order):
        output = self.transformer_decoder(pos_target_tensor)
        logit = self.hidden2logit[f"layer_{feature}"](output[:, -1:])
        logits_dict[feature] = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        input_seq_list.append(feature_emb_reshape)
        input_seq_tensor = torch.cat(input_seq_list, dim=1)
        pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='last')
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(input_seq_list, target) # (B*T) x (window_size + BOS + num_sub_tokens-1) x d_model
    pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS + num_sub_tokens-1) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class SelfAttentionUniAudio(SelfAttention):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    '''
    Uniaudio version of self-attention sub-decoder
    Through the experiments, we found that the performance of the model is better than our proposed self-attention sub-decoder
    It shows comparable performance with the cross-attention sub-decoder
    However, NMT shows better performance than UniAudio in terms of the performance of the model
    '''

  def _prepare_token_embedding_for_teacher_forcing(self, hidden_vec_reshape, target):
    input_seq_list = []
    # append zero vector
    input_seq_list.append(torch.zeros(hidden_vec_reshape.shape[0], 1, hidden_vec_reshape.shape[2]).to(self.device))
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq_list.append(feature_emb_reshape)

    feature_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x num_sub-tokens x d_model
    # Ensure hidden_vec_reshape and feature_tensor have the same shape
    assert hidden_vec_reshape.shape == feature_tensor.shape, f"Shapes of hidden_vec_reshape and feature_tensor do not match: {hidden_vec_reshape.shape} vs {feature_tensor.shape}"
    # Sum hidden_vec_reshape and feature_tensor in the last dimension
    memory_tensor = hidden_vec_reshape + feature_tensor
    return memory_tensor
  
  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x num_sub-tokens
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    hidden_vec_reshape = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub-tokens x d_model
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      pos_target_tensor = self._apply_pos_enc(hidden_vec_reshape, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      for idx, feature in enumerate(self.prediction_order):
        output = self.transformer_decoder(pos_target_tensor)
        logit = self.hidden2logit[f"layer_{feature}"](output[:, -1:])
        logits_dict[feature] = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        pos_target_tensor = torch.cat([pos_target_tensor[:, :idx+1, :], feature_emb_reshape + pos_target_tensor[:, idx+1:idx+2, :], pos_target_tensor[:, idx+2:, :]], dim=1)

      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(hidden_vec_reshape, target) # (B*T) x (window_size + BOS + num_sub_tokens-1) x d_model
    pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS + num_sub_tokens-1) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class CrossAttention(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    The power of Cross-attention and UniAudio style Self-attention lies in that using the output of the main decoder or hidden vec directly in the sub-decoder
    As the output of the main decoder is the representation of the whole sequence, 
    it contains richer information which can even decode out sub-tokens in a parallel manner
    So both architectures using the output of the main decoder in a direct way show better performance than the original self-attention sub-decoder
    '''
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.sub_decoder_enricher_use = sub_decoder_enricher_use
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(len(self.prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)

    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    if sub_decoder_enricher_use:
      self.enricher_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    causal_mask = generate_SA_mask(len(prediction_order))
    causl_ca_mask = generate_CA_mask(len(prediction_order), len(prediction_order)).to(self.device)
    self.register_buffer('causal_mask', causal_mask)
    self.register_buffer('causal_ca_mask', causl_ca_mask)

    if sub_decoder_depth > 1:
      self.sub_decoder_layers = nn.Sequential(
        *[DecoderLayer(dim=dim, num_heads=heads, dropout=dropout) for _ in range(sub_decoder_depth)]
      )
    else:
      self.sub_decoder_layers = nn.Sequential(DecoderLayer(dim=dim, num_heads=heads, dropout=dropout))
    if sub_decoder_enricher_use:
      self.feature_enricher_layers = nn.Sequential(FeatureEnricher(dim=dim, num_heads=heads, dropout=dropout))

  def _apply_window_on_hidden_vec(self, hidden_vec):
    BOS_emb = self.enricher_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    # through our experiments, we found that the size of the window doesn't affect the performance of the model much
    window_size = 1
    zero_vec = torch.zeros((hidden_vec.shape[0], window_size-1, hidden_vec.shape[2])).to(self.device) # B x (window_size-1) x d_model
    cat_hidden_vec = torch.cat([zero_vec, hidden_vec], dim=1) # B x (window_size-1+T) x d_model
    new_hidden_vec = cat_hidden_vec.unfold(1, window_size, 1).transpose(2, 3) # B x T x window_size x d_model
    new_hidden_vec = new_hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], window_size, -1)) # (B*T) x window_size x d_model
    new_hidden_vec = torch.cat([BOS_emb, new_hidden_vec], dim=1) # (B*T) x (window_size+1) x d_model
    return new_hidden_vec

  def _apply_pos_enc(self, tgt):
    pos = torch.arange(tgt.shape[1]).to(tgt.device) # num_sub_tokens
    pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1) # (B*T) x num_sub_tokens
    tgt_pos = tgt + self.pos_enc(pos.long()) # (B*T) x num_sub_tokens x d_model
    return tgt_pos

  def _prepare_token_embedding_for_teacher_forcing(self, memory_list, target):
    for _, feature in enumerate(self.prediction_order[:-1]):
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      memory_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + num_sub_tokens-1) x d_model
    return memory_tensor

  def _prepare_memory_list(self, hidden_vec, target=None):
    memory_list = [] # used for key and value in cross attention
    BOS_emb = self.sub_decoder_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    if target is not None: # training
      memory_list.append(BOS_emb)
    else: # inference
      memory_list.append(BOS_emb[-1:, :, :])
    return memory_list

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target']

    # apply window on hidden_vec for enricher
    if self.sub_decoder_enricher_use:
      window_applied_hidden_vec = self._apply_window_on_hidden_vec(hidden_vec) # (B*T) x window_size x d_model
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub_tokens x d_model
    input_seq_pos = self._apply_pos_enc(input_seq)
    # prepare memory
    memory_list = self._prepare_memory_list(hidden_vec=hidden_vec, target=target)
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x 1 x d_model
      old_memory_tensor = memory_tensor
      for idx, feature in enumerate(self.prediction_order):
        feature_pos = self.feature_order_in_output[feature]
        if self.sub_decoder_enricher_use:
          input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec[-1:]}
          input_dict = self.feature_enricher_layers(input_dict)
          memory_tensor = input_dict['input_seq']
        CA_attn_mask = generate_CA_mask(input_seq_pos.shape[1], memory_tensor.shape[1]).to(self.device)
        input_dict = {'input_seq': input_seq_pos[-1:], 'memory': memory_tensor, 'memory_mask': CA_attn_mask}
        input_dict = self.sub_decoder_layers(input_dict)
        attn_output = input_dict['input_seq']
        logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
        logit = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        logits_dict[feature] = logit
        sampled_token,prob = sample_with_prob(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        memory_list.append(feature_emb_reshape)
        memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + idx+1) x d_model
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    memory_tensor = self._prepare_token_embedding_for_teacher_forcing(memory_list, target) # (B*T) x (BOS + num_sub_tokens-1) x d_model
    # apply feature enricher to memory
    if self.sub_decoder_enricher_use:
      input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
      input_dict = self.feature_enricher_layers(input_dict)
      memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # implement sub decoder cross attention
    input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
    input_dict = self.sub_decoder_layers(input_dict)
    attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # get prob
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class Flatten4Encodec(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    hidden_vec = input_dict['hidden_vec']

    # ---- Training ---- #
    logits_tensor = torch.zeros(hidden_vec.shape[0], hidden_vec.shape[1], 2049).to(self.device)
    for idx, feature_type in enumerate(self.prediction_order):
      # ::4 means that we only use the first token in each 4 tokens
      # so the chosen tokens will be: 0, 4, 8, 12, ...
      # 1::4 means that we only use the second token in each 4 tokens
      # so the chosen tokens will be: 1, 5, 9, 13, ...
      separated_hidden_vec = hidden_vec[:, idx::4, :]
      logit = self.hidden2logit[f"layer_{feature_type}"](separated_hidden_vec)
      logits_tensor[:, idx::4, :] = logit
      # prob_dict[feature_type] = prob
    return logits_tensor
  
  def run_one_step(self, input_dict, sampling_method=None, threshold=None, temperature=None, feature_type=None):
    # ---- Generate(Inference) ---- #
    hidden_vec = input_dict['hidden_vec']
    logit = self.hidden2logit[f"layer_{feature_type}"](hidden_vec[:, -1:])
    sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    return logit, sampled_token


class DiffusionDecoder(SubDecoderClass):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool,
    MASK_IDX:int = 126336,
    denoising_steps:int = 8,
    eps:float = 1e-3,
    method:str = 'low-confidence', # or random or auto-regressive
  ):
    '''
    The power of Cross-attention and UniAudio style Self-attention lies in that using the output of the main decoder or hidden vec directly in the sub-decoder
    As the output of the main decoder is the representation of the whole sequence, 
    it contains richer information which can even decode out sub-tokens in a parallel manner
    So both architectures using the output of the main decoder in a direct way show better performance than the original self-attention sub-decoder
    '''
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.sub_decoder_enricher_use = sub_decoder_enricher_use
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(len(self.prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)

    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    self.diffusion_mask_emb = nn.Parameter(torch.empty(dim), requires_grad=True) # embedding of mask token,idx is 126336,which is not in vocab
    nn.init.normal_(self.diffusion_mask_emb, mean=0.0, std=0.02)
    self.MASK_idx = MASK_IDX
    self.denoising_steps = denoising_steps
    self.eps = eps
    self.method = method
    
    self.input_norm = nn.LayerNorm(dim)
    
    self.feature_boost_layers = nn.Sequential(TransformerLayer(dim=dim, num_heads=heads, dropout=dropout))
    
    if sub_decoder_enricher_use:
      self.enricher_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    causal_mask = generate_SA_mask(len(prediction_order))
    causal_ca_mask = generate_none_causality_mask(len(prediction_order), len(prediction_order)).to(self.device)
    self.register_buffer('causal_mask', causal_mask)
    self.register_buffer('causal_ca_mask', causal_ca_mask)
    
    # get depth of the sub-decoder
    if sub_decoder_depth > 1:
      self.sub_decoder_layers = nn.Sequential(*[TransformerLayer(dim=dim, num_heads=heads, dropout=dropout) for _ in range(sub_decoder_depth)])
    else:
      self.sub_decoder_layers = nn.Sequential(TransformerLayer(dim=dim, num_heads=heads, dropout=dropout))
    if sub_decoder_enricher_use:
      self.feature_enricher_layers = nn.Sequential(FeatureEnricher(dim=dim, num_heads=heads, dropout=dropout))

  
  # simplified version of the forward process in diffusion model
  def _forward_process(self, input_ids, eps=1e-3, mask_idx=None):
    reshaped_input_ids = torch.reshape(input_ids, (-1, input_ids.shape[-1])) # B*T x num_sub_tokens
    b, l = reshaped_input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token,attention that this token is not in the vocab
    if mask_idx is not None:
      noisy_batch = torch.where(masked_indices, mask_idx, reshaped_input_ids)
    else:
      noisy_batch = torch.where(masked_indices, 126336, reshaped_input_ids)# 126336 is used for [MASK] token in
    return noisy_batch, masked_indices, p_mask
  
    
  def _apply_window_on_hidden_vec(self, hidden_vec):
    BOS_emb = self.enricher_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    # through our experiments, we found that the size of the window doesn't affect the performance of the model much
    window_size = 1
    zero_vec = torch.zeros((hidden_vec.shape[0], window_size-1, hidden_vec.shape[2])).to(self.device) # B x (window_size-1) x d_model
    cat_hidden_vec = torch.cat([zero_vec, hidden_vec], dim=1) # B x (window_size-1+T) x d_model
    new_hidden_vec = cat_hidden_vec.unfold(1, window_size, 1).transpose(2, 3) # B x T x window_size x d_model
    new_hidden_vec = new_hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], window_size, -1)) # (B*T) x window_size x d_model
    new_hidden_vec = torch.cat([BOS_emb, new_hidden_vec], dim=1) # (B*T) x (window_size+1) x d_model
    return new_hidden_vec

  def _apply_pos_enc(self, tgt):
    pos = torch.arange(tgt.shape[1]).to(tgt.device) # num_sub_tokens
    pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1) # (B*T) x num_sub_tokens
    tgt_pos = tgt + self.pos_enc(pos.long()) # (B*T) x num_sub_tokens x d_model
    return tgt_pos

  def _prepare_token_embedding_for_teacher_forcing(self, memory_list, target):
    for _, feature in enumerate(self.prediction_order[:-1]):
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      memory_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + num_sub_tokens-1) x d_model
    return memory_tensor

  # return a tensor 
  def _get_noisy_tensor(self, target_shape):
    new_target = torch.zeros(target_shape).to(self.device)
    # fill all the elements in the tensor with the embedding of the mask token
    new_target[:, :, :] = self.diffusion_mask_emb
    return new_target
    
    # prepare the embedding of the target,
  def _prepare_embedding(self, memory_list, target):
    for _, feature in enumerate(self.prediction_order):
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      memory_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + num_sub_tokens) x d_model
    return memory_tensor

      
  def _prepare_memory_list(self, hidden_vec, target=None, add_BOS=True):
    memory_list = [] # used for key and value in cross attention
    BOS_emb = self.sub_decoder_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    if add_BOS is true:
      if target is not None: # training
        memory_list.append(BOS_emb)
      else: # inference
        memory_list.append(BOS_emb[-1:, :, :])
    else:
      pass
    return memory_list

  def _get_num_transfer_tokens(self, mask_index, steps):
      '''
      In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
      Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
      the expected number of tokens transitioned at each step should be consistent.

      This function is designed to precompute the number of tokens that need to be transitioned at each step.
      '''
      mask_num = mask_index.sum(dim=1, keepdim=True)
      base = mask_num // steps
      remainder = mask_num % steps

      num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

      for i in range(mask_num.size(0)):
          num_transfer_tokens[i, :remainder[i]] += 1

      return num_transfer_tokens
    
  def sample_from_logits(self, attn_output, hidden_vec, sampling_method=None, threshold=None, temperature=None, force_decode=False,step=None):
    sampled_token_dict = {}
    logits_dict = {}
    candidate_token_embeddings = {}
    candidate_token_probs = {}
    b,t,d = hidden_vec.shape # B x T x d_model
    # print("*"*8)
    logits_list = []
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_list.append(logit)
    for idx, feature in enumerate(self.prediction_order):
      logit = logits_list[idx] # B x T x vocab_siz
      sampled_token, prob = sample_with_prob(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      if step==0 and force_decode:
        if feature == 'velocity':
          sampled_token = torch.tensor([2]).to(logit.device)
          prob = torch.tensor([1.0]).to(logit.device)
        else:
          prob = torch.tensor([0.0]).to(logit.device)
          # print(feature, sampled_token, prob)
      sampled_token_dict[feature] = sampled_token
      logits_dict[feature] = logit
      candidate_token_probs[feature] = prob
      feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
      feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
      candidate_token_embeddings[feature] = feature_emb_reshape
    stacked_logits_probs = torch.stack(list(candidate_token_probs.values()), dim=0).reshape((b*t, -1)) # (B*T) x num_sub_tokens x vocab_size
    stacked_token_embeddings = torch.stack(list(candidate_token_embeddings.values()), dim=0).reshape((b*t, -1, d)) # (B*T) x num_sub_tokens x d_model
    # print("sampled_token_dict", sampled_token_dict)
    return sampled_token_dict, logits_dict, candidate_token_probs, stacked_logits_probs, stacked_token_embeddings

  def sample_from_logits_fast(self, attn_output, hidden_vec, sampling_method=None, threshold=None, temperature=None):
    sampled_token_dict = {}
    logits_dict = {}
    candidate_token_embeddings = {}
    candidate_token_probs = {}

    b, t, d = hidden_vec.shape  # (B, T, D)
    F = len(self.projection_keys)
    Vmax = self.max_vocab_size

    # === 1. 取出所有 feature 的位置 === #
    feature_pos_list = [self.feature_order_in_output[f] for f in self.projection_keys]

    # === 2. 提取 attn_output 中各 feature 的位置 → (B, F, D) === #
    attn_features = torch.stack(
        [attn_output[:, pos, :] for pos in feature_pos_list], dim=1
    )  # (B, F, D)

    # === 3. 使用 batch 矩阵乘法：einsum 实现并行 Linear === #
    # attn_features: (B, F, D)
    # proj_weight:   (F, Vmax, D)
    # proj_bias:     (F, Vmax)
    # output: (B, F, Vmax)
    logits = torch.einsum("bfd,fvd->bfv", attn_features, self.proj_weight) + self.proj_bias

    # === 4. 按照原始 vocab size 截断每个 feature 的 logits === #
    logits_list = []
    logits_dict_by_feature = {
    feature: logits[:, i, :self.vocab_sizes[feature]]
    for i, feature in enumerate(self.projection_keys)
}
    for i, feature in enumerate(self.projection_keys):
        vocab_size = self.vocab_sizes[feature]
        logits_list.append(logits[:, i, :vocab_size])  # (B, vocab_size)
    for idx, feature in enumerate(self.prediction_order):
      logit = logits_dict_by_feature[feature].unsqueeze(0)  # B x T x vocab_size  
      sampled_token, prob = sample_with_prob_fast(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      # print(feature, sampled_token, prob)
      sampled_token_dict[feature] = sampled_token.squeeze(0)  # B x T
      logits_dict[feature] = logit
      candidate_token_probs[feature] = prob
      feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
      feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
      candidate_token_embeddings[feature] = feature_emb_reshape
    stacked_logits_probs = torch.stack(list(candidate_token_probs.values()), dim=0).reshape((b*t, -1)) # (B*T) x num_sub_tokens x vocab_size
    stacked_token_embeddings = torch.stack(list(candidate_token_embeddings.values()), dim=0).reshape((b*t, -1, d)) # (B*T) x num_sub_tokens x d_model

    return sampled_token_dict, logits_dict, candidate_token_probs, stacked_logits_probs, stacked_token_embeddings

  def choose_tokens(self, hidden_vec, step, method, stacked_logits_probs, num_transfer_tokens):
    if method == 'low-confidence':
      _, indices = torch.topk(stacked_logits_probs, k=int(num_transfer_tokens[:,step]), dim=-1)
    elif method == 'random':
      indices = torch.randint(0, stacked_logits_probs.shape[-1], (num_transfer_tokens[:, step],)).to(logit.device)
    elif method == 'auto-regressive':
      indices = torch.tensor([[step]], device=hidden_vec.device)
    return indices
  
  
  def forward_(self, input_dict, sampling_method=None, threshold=None, temperature=None, worst_case=False, validation=False):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] #B x T x d_model
    

    # apply window on hidden_vec for enricher
    if self.sub_decoder_enricher_use:
      window_applied_hidden_vec = self._apply_window_on_hidden_vec(hidden_vec) # (B*T) x window_size x d_model
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub_tokens x d_model
    input_seq_pos = input_seq
    # input_seq_pos = self._apply_pos_enc(input_seq) # (B*T) x num_sub_tokens x d_model
    # prepare memory
    memory_list = self._prepare_memory_list(hidden_vec=hidden_vec, target=target, add_BOS=False)
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      b,t,d = hidden_vec.shape # B x T x d_model
      l = len(self.prediction_order) # num_sub_tokens
      memory_tensor = self._get_noisy_tensor(target_shape=(b*t, l, d))
      all_noise_tensor = memory_tensor.clone() # (B*T) x num_sub_tokens x d_model
      
      # indicate the position of the mask token,1 means that the token hsa been masked
      masked_history = torch.ones((b*t, l), device=hidden_vec.device, dtype=torch.int64).bool()
      num_transfer_tokens = self._get_num_transfer_tokens(masked_history, self.denoising_steps)
      # denoising c
      stored_logits_dict = {}
      stored_probs_dict = {}
      for step in range(self.denoising_steps):
        # nomalize the memory tensor
        # memory_tensor = self.layer_norm(memory_tensor) # (B*T) x num_sub_tokens x d_model
        if self.sub_decoder_enricher_use:
          input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
          input_dict = self.feature_enricher_layers(input_dict)
          memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
        input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
        # input_dict = {'input_seq': memory_tensor, 'memory': input_seq_pos, 'memory_mask': self.causal_ca_mask}
        input_dict = self.sub_decoder_layers(input_dict)
        attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
        candidate_token_probs = {}
        sampled_token_dict, logits_dict, candidate_token_probs, stacked_logits_probs, stacked_token_embeddings = self.sample_from_logits(attn_output, hidden_vec, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        
        # set prob of the changed tokens to -inf
        stacked_logits_probs = torch.where(masked_history, stacked_logits_probs, -torch.inf)
        # indices = self.choose_tokens(hidden_vec,step, "auto-regressive", stacked_logits_probs, num_transfer_tokens)
        indices = self.choose_tokens(hidden_vec, step, self.method, stacked_logits_probs, num_transfer_tokens)  
        # breakpoint()
        # undate the masked history
        for i in range(b*t):
          for j in range(l):
            if j in indices[i]:
              masked_history[i][j] = False
              stored_logits_dict[self.prediction_order[j]] = logits_dict[self.prediction_order[j]].clone()
              stored_probs_dict[self.prediction_order[j]] = candidate_token_probs[self.prediction_order[j]].clone()
        expand_masked_history = masked_history.unsqueeze(-1).expand(-1, -1, memory_tensor.shape[-1]) # (B*T) x num_sub_tokens x d_model
        memory_tensor = torch.where(expand_masked_history, all_noise_tensor, stacked_token_embeddings)
      # breakpoint()
      # print("stored_probs_dict", stored_probs_dict)
      # print("sampled_token_dict", sampled_token_dict)
      return stored_logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    _, masked_indices, p_mask = self._forward_process(target, mask_idx=self.MASK_idx) # (B*T) x (num_sub_tokens) x d_model
    memory_tensor = self._prepare_embedding(memory_list, target) # (B*T) x (num_sub_tokens) x d_model
    # apply layer norm
   
    extend_masked_indices = masked_indices.unsqueeze(-1).expand(-1, -1, memory_tensor.shape[-1]) # (B*T) x (num_sub_tokens) x d_model
    if worst_case: # mask all ,turn into parallel
      extend_masked_indices = torch.ones_like(extend_masked_indices).to(self.device)
    memory_tensor = torch.where(extend_masked_indices, self.diffusion_mask_emb, memory_tensor)
    if self.sub_decoder_enricher_use:
      input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
      input_dict = self.feature_enricher_layers(input_dict)
      memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
    input_dict = self.sub_decoder_layers(input_dict)
    attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # get prob
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict, (masked_indices, p_mask)
  
  def forward_old(self, input_dict, sampling_method=None, threshold=None, temperature=None, worst_case=False, validation=False):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] #B x T x d_model
    bos_hidden_vec = input_dict['bos_token_hidden'] # B x 1 x d_model, used for the first token in the sub-decoder

    # apply window on hidden_vec for enricher
    if self.sub_decoder_enricher_use:
      window_applied_hidden_vec = self._apply_window_on_hidden_vec(hidden_vec) # (B*T) x window_size x d_model
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub_tokens x d_model
    input_seq_pos = self._apply_pos_enc(input_seq) # (B*T) x num_sub_tokens x d_model
    
    if bos_hidden_vec is None: # start of generation
      if target is None:
        bos_hidden_vec = input_seq_pos
      else:
        bos_hidden_vec =hidden_vec[:, 0, :].unsqueeze(1).repeat(1, hidden_vec.shape[1], 1) # B x T x d_model
        bos_hidden_vec = bos_hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1))
        bos_hidden_vec = bos_hidden_vec.repeat(1, len(self.prediction_order), 1)
        
    else:
      bos_hidden_vec = bos_hidden_vec.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub_tokens x d_model
    
    # input_seq_pos = input_seq
    input_dict = {'input_seq': input_seq_pos, 'memory': bos_hidden_vec, 'memory_mask': self.causal_ca_mask}
    boosted_input_dict = self.feature_boost_layers(input_dict) # (B*T) x num_sub_tokens x d_model
    input_seq_pos = boosted_input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # input_seq_pos = self.input_norm(input_seq_pos) # (B*T) x num_sub_tokens x d_model
    # input_seq_pos = self._apply_pos_enc(input_seq) # (B*T) x num_sub_tokens x d_model
    # prepare memory
    memory_list = self._prepare_memory_list(hidden_vec=hidden_vec, target=target, add_BOS=False)
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      b,t,d = hidden_vec.shape # B x T x d_model
      l = len(self.prediction_order) # num_sub_tokens
      memory_tensor = self._get_noisy_tensor(target_shape=(b*t, l, d))
      all_noise_tensor = memory_tensor.clone() # (B*T) x num_sub_tokens x d_model
      
      # indicate the position of the mask token,1 means that the token hsa been masked
      masked_history = torch.ones((b*t, l), device=hidden_vec.device, dtype=torch.int64).bool()
      num_transfer_tokens = self._get_num_transfer_tokens(masked_history, self.denoising_steps)
      # denoising c
      stored_logits_dict = {}
      stored_probs_dict = {}
      for step in range(self.denoising_steps):
        memory_tensor = self._apply_pos_enc(memory_tensor) # (B*T) x num_sub_tokens x d_model
        # nomalize the memory tensor
        # memory_tensor = self.layer_norm(memory_tensor) # (B*T) x num_sub_tokens x d_model
        if self.sub_decoder_enricher_use:
          input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
          input_dict = self.feature_enricher_layers(input_dict)
          memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
        # input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
        input_dict = {'input_seq': memory_tensor, 'memory': input_seq_pos, 'memory_mask': self.causal_ca_mask}
        input_dict = self.sub_decoder_layers(input_dict)
        attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
        candidate_token_probs = {}
        candidate_token_embeddings = {}
        for idx, feature in enumerate(self.prediction_order):
          feature_pos = self.feature_order_in_output[feature]
          logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
          logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
          logits_dict[feature] = logit
          sampled_token,probs = sample_with_prob(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
          # print(idx,feature,sampled_token,probs)
          sampled_token_dict[feature] = sampled_token
          candidate_token_probs[feature] = probs
          feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
          feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
          candidate_token_embeddings[feature] = feature_emb_reshape

        stacked_logits_probs = torch.stack(list(candidate_token_probs.values()), dim=0).reshape((b*t, l)) # (B*T) x num_sub_tokens x vocab_size
        stacked_token_embeddings = torch.stack(list(candidate_token_embeddings.values()), dim=0).reshape((b*t, l, d))
        
        # set prob of the changed tokens to -inf
        stacked_logits_probs = torch.where(masked_history, stacked_logits_probs, -torch.inf)

        if self.method == 'low-confidence':
          _, indices = torch.topk(stacked_logits_probs, k=int(num_transfer_tokens[:,step]), dim=-1)
        elif self.method == 'random':
          indices = torch.randint(0, stacked_logits_probs.shape[-1], (num_transfer_tokens[:, step],)).to(logit.device)
        elif self.method == 'auto-regressive':
          indices = torch.tensor([[step]], device=logit.device)
        # undate the masked history
        for i in range(b*t):
          for j in range(l):
            if j in indices[i]:
              masked_history[i][j] = False
              stored_logits_dict[self.prediction_order[j]] = logits_dict[self.prediction_order[j]].clone()
              stored_probs_dict[self.prediction_order[j]] = candidate_token_probs[self.prediction_order[j]].clone()
        expand_masked_history = masked_history.unsqueeze(-1).expand(-1, -1, memory_tensor.shape[-1]) # (B*T) x num_sub_tokens x d_model
        memory_tensor = torch.where(expand_masked_history, all_noise_tensor, stacked_token_embeddings)
      return stored_logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    _, masked_indices, p_mask = self._forward_process(target, mask_idx=self.MASK_idx) # (B*T) x (num_sub_tokens) x d_model
    memory_tensor = self._prepare_embedding(memory_list, target) # (B*T) x (num_sub_tokens) x d_model
    # apply layer norm
   
    extend_masked_indices = masked_indices.unsqueeze(-1).expand(-1, -1, memory_tensor.shape[-1]) # (B*T) x (num_sub_tokens) x d_model
    if worst_case: # mask all ,turn into parallel
      extend_masked_indices = torch.ones_like(extend_masked_indices).to(self.device)
    memory_tensor = torch.where(extend_masked_indices, self.diffusion_mask_emb, memory_tensor)
    memory_tensor = self._apply_pos_enc(memory_tensor) # (B*T) x num_sub_tokens x d_model
    # all is embedding
    # memory_tensor = self.layer_norm(memory_tensor)
    # apply feature enricher to memory
    if self.sub_decoder_enricher_use:
      input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
      input_dict = self.feature_enricher_layers(input_dict)
      memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # implement sub decoder cross attention
    # input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
    # inter_input = torch.cat([input_seq_pos, memory_tensor], dim=1)
    # inter_input = input_seq_pos + memory_tensor # (B*T) x num_sub_tokens x d_model
    # input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
    input_dict = {'input_seq': memory_tensor, 'memory': input_seq_pos, 'memory_mask': self.causal_ca_mask}
    input_dict = self.sub_decoder_layers(input_dict)
    attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # get prob
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict, (masked_indices, p_mask)
  
  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None, Force_decode=False, worst_case=False, validation=False):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] #B x T x d_model
    bos_hidden_vec = input_dict['bos_token_hidden'] # B x 1 x d_model, used for the first token in the sub-decoder

    # apply window on hidden_vec for enricher
    if self.sub_decoder_enricher_use:
      window_applied_hidden_vec = self._apply_window_on_hidden_vec(hidden_vec) # (B*T) x window_size x d_model
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub_tokens x d_model
    input_seq_pos = self._apply_pos_enc(input_seq) # (B*T) x num_sub_tokens x d_model
    
    if bos_hidden_vec is None: # start of generation
      if target is None:
        bos_hidden_vec = input_seq_pos
      else:
        bos_hidden_vec =hidden_vec[:, 0, :].unsqueeze(1).repeat(1, hidden_vec.shape[1], 1) # B x T x d_model
        bos_hidden_vec = bos_hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1))
        bos_hidden_vec = bos_hidden_vec.repeat(1, len(self.prediction_order), 1)
        
    else:
      bos_hidden_vec = bos_hidden_vec.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub_tokens x d_model
    
    # input_seq_pos = input_seq
    input_dict = {'input_seq': input_seq_pos, 'memory': bos_hidden_vec, 'memory_mask': self.causal_ca_mask}
    boosted_input_dict = self.feature_boost_layers(input_dict) # (B*T) x num_sub_tokens x d_model
    input_seq_pos = boosted_input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # input_seq_pos = self.input_norm(input_seq_pos) # (B*T) x num_sub_tokens x d_model
    # input_seq_pos = self._apply_pos_enc(input_seq) # (B*T) x num_sub_tokens x d_model
    # prepare memory
    memory_list = self._prepare_memory_list(hidden_vec=hidden_vec, target=target, add_BOS=False)
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      b,t,d = hidden_vec.shape # B x T x d_model
      l = len(self.prediction_order) # num_sub_tokens
      memory_tensor = self._get_noisy_tensor(target_shape=(b*t, l, d))
      all_noise_tensor = memory_tensor.clone() # (B*T) x num_sub_tokens x d_model
      
      # indicate the position of the mask token,1 means that the token hsa been masked
      masked_history = torch.ones((b*t, l), device=hidden_vec.device, dtype=torch.int64).bool()
      num_transfer_tokens = self._get_num_transfer_tokens(masked_history, self.denoising_steps)
      # denoising c
      stored_logits_dict = {}
      stored_probs_dict = {}
  #     with torch.profiler.profile(
  #     activities=[
  #         torch.profiler.ProfilerActivity.CPU,
  #         torch.profiler.ProfilerActivity.CUDA],
  #     record_shapes=True,
  #     profile_memory=True,
  #     with_stack=True
  # ) as prof:
      for step in range(self.denoising_steps):
        memory_tensor = self._apply_pos_enc(memory_tensor) # (B*T) x num_sub_tokens x d_model
        # nomalize the memory tensor
        # memory_tensor = self.layer_norm(memory_tensor) # (B*T) x num_sub_tokens x d_model
        if self.sub_decoder_enricher_use:
          input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
          input_dict = self.feature_enricher_layers(input_dict)
          memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
        # input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
        input_dict = {'input_seq': memory_tensor, 'memory': input_seq_pos, 'memory_mask': self.causal_ca_mask}
        input_dict = self.sub_decoder_layers(input_dict)
        attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
        candidate_token_probs = {}
        
        sampled_token_dict, logits_dict, candidate_token_probs, stacked_logits_probs, stacked_token_embeddings = self.sample_from_logits(attn_output, hidden_vec, sampling_method=sampling_method, threshold=threshold, temperature=temperature,
                                                                                                                                         force_decode=Force_decode,
                                                                                                                                         step=step)

        # set prob of the changed tokens to -inf
        stacked_logits_probs = torch.where(masked_history, stacked_logits_probs, -torch.inf)

        if self.method == 'low-confidence':
          _, indices = torch.topk(stacked_logits_probs, k=int(num_transfer_tokens[:,step]), dim=-1)
        elif self.method == 'random':
          indices = torch.randint(0, stacked_logits_probs.shape[-1], (num_transfer_tokens[:, step],)).to(logit.device)
        elif self.method == 'auto-regressive':
          indices = torch.tensor([[step]], device=logit.device)
        # undate the masked history
        for i in range(b*t):
          for j in range(l):
            if j in indices[i]:
              masked_history[i][j] = False
              stored_logits_dict[self.prediction_order[j]] = logits_dict[self.prediction_order[j]].clone()
        expand_masked_history = masked_history.unsqueeze(-1).expand(-1, -1, memory_tensor.shape[-1]) # (B*T) x num_sub_tokens x d_model
        memory_tensor = torch.where(expand_masked_history, all_noise_tensor, stacked_token_embeddings)
      # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
      # print(sampled_token_dict)
      return stored_logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    _, masked_indices, p_mask = self._forward_process(target, mask_idx=self.MASK_idx) # (B*T) x (num_sub_tokens) x d_model
    memory_tensor = self._prepare_embedding(memory_list, target) # (B*T) x (num_sub_tokens) x d_model
    # apply layer norm
   
    extend_masked_indices = masked_indices.unsqueeze(-1).expand(-1, -1, memory_tensor.shape[-1]) # (B*T) x (num_sub_tokens) x d_model
    if worst_case: # mask all ,turn into parallel
      extend_masked_indices = torch.ones_like(extend_masked_indices).to(self.device)
    memory_tensor = torch.where(extend_masked_indices, self.diffusion_mask_emb, memory_tensor)
    memory_tensor = self._apply_pos_enc(memory_tensor) # (B*T) x num_sub_tokens x d_model
    # all is embedding
    # memory_tensor = self.layer_norm(memory_tensor)
    # apply feature enricher to memory
    if self.sub_decoder_enricher_use:
      input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
      input_dict = self.feature_enricher_layers(input_dict)
      memory_tensor = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # implement sub decoder cross attention
    input_dict = {'input_seq': memory_tensor, 'memory': input_seq_pos, 'memory_mask': self.causal_ca_mask}
    input_dict = self.sub_decoder_layers(input_dict)
    attn_output = input_dict['input_seq'] # (B*T) x num_sub_tokens x d_model
    # get prob
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict, (masked_indices, p_mask)