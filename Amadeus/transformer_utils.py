import torch
import torch.nn as nn

from x_transformers import Decoder, Encoder, PrefixDecoder, CrossAttender
from transformers import T5EncoderModel
from data_representation.vocab_utils import LangTokenVocab
  
class PosEncoding(nn.Module):
  def __init__(self, emb_size, max_t):
    super().__init__()
    self.emb_size =emb_size
    self.max_t = max_t
    self.register_buffer('encoding', self._prepare_emb())

  def _prepare_emb(self):
    dim_axis = 10000**(torch.arange(self.emb_size//2) * 2 / self.emb_size) # 10000 ** (normalized values between 0~1 num_emb_dim)
    timesteps = torch.arange(self.max_t)
    pos_enc_in = timesteps.unsqueeze(1) / dim_axis.unsqueeze(0)
    pos_enc_sin = torch.sin(pos_enc_in) # x values for sin are between 0 ~ 1 so the values could never be the same
    pos_enc_cos = torch.cos(pos_enc_in)

    pos_enc = torch.stack([pos_enc_sin, pos_enc_cos], dim=-1).reshape([self.max_t, self.emb_size])
    return pos_enc

  def forward(self, x):
    return self.encoding[x]

class ResidualLayerNormModule(nn.Module):
  def __init__(self, submodule):
    super().__init__()
    self.submodule = submodule
    self.layer_norm = nn.LayerNorm(self.submodule.input_size)

  def forward(self, x, mask=None, y=None):
    if y is not None:
      res_x = self.submodule(x, y, mask)
    elif mask is not None:
      res_x = self.submodule(x, mask)
    else:
      res_x = self.submodule(x)
    x =  x + res_x
    return self.layer_norm(x)

class SingleEmbedding(nn.Module):
  def __init__(
    self, 
    vocab, 
    dim_model,
  ):
    '''
    Embedding layer for REMI
    '''
    super().__init__()
    vocab_size = vocab.get_vocab_size()
    self.embedding = nn.Embedding(vocab_size, dim_model)

  def forward(self, x):
    return self.embedding(x)

class MultiEmbedding(nn.Module):
  def __init__(
    self, 
    vocab:LangTokenVocab,
    dim_model:int,
  ):
    super().__init__()
    '''
    Embedding layer for compound tokens
    '''
    self.vocab_size = vocab.get_vocab_size()
    self.feature_list = vocab.feature_list
    self.dim_model = dim_model
    self.layers = []

    self._make_emb_layers()
    self._init_params()
    self._make_emb_boundaries_by_key()
  
  def _init_params(self):
    # apply kaiming init
    for layer in self.layers:
      if isinstance(layer, nn.Embedding):
        nn.init.kaiming_normal_(layer.weight)

  def _make_emb_layers(self):
    vocab_sizes = [self.vocab_size[key] for key in self.feature_list]
    self.embedding_sizes = [self.dim_model for _ in self.feature_list]
    for vocab_size, embedding_size in zip(vocab_sizes, self.embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def _make_emb_boundaries_by_key(self):
    '''
    This function returns dict of boundaries for each embedding layer
    '''
    self.emb_boundary_by_key = {}
    start_idx = 0
    for key, emb_size in zip(self.feature_list, self.embedding_sizes):
      if emb_size != 0:
        self.emb_boundary_by_key[key] = (start_idx, start_idx + emb_size)
        start_idx += emb_size

  def forward(self, x):
    emb = torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)
    return emb

  def __len__(self):
    return len(self.layers)

  def get_emb_by_key(self, key, token):
    layer_idx = self.feature_list.index(key)
    return self.layers[layer_idx](token)

class SummationEmbedder(MultiEmbedding):
  def __init__(
    self, 
    vocab:LangTokenVocab, 
    dim_model:int
  ):
    super().__init__(vocab, dim_model)

  def forward(self, seq):
    emb_list = [module(seq[..., i]) for i, module in enumerate(self.layers)]
    stacked_emb = torch.stack(emb_list, dim=2) # B x T x num_features x emb_size
    output = torch.sum(stacked_emb, dim=2) # B x T x emb_size
    return output

class AverageEmbedder(MultiEmbedding):
  def __init__(
    self, 
    vocab:LangTokenVocab, 
    dim_model:int
  ):
    super().__init__(vocab, dim_model)

  def forward(self, seq):
    emb_list = [module(seq[..., i]) for i, module in enumerate(self.layers)]
    stacked_emb = torch.stack(emb_list, dim=2) # B x T x num_features x emb_size
    output = torch.mean(stacked_emb, dim=2) # B x T x emb_size
    return output

class SelfAttentionEmbedder(MultiEmbedding):
  def __init__(
    self, 
    vocab:LangTokenVocab, 
    dim_model:int
  ):
    super().__init__(vocab, dim_model)
    self.dropout = 0.1

    self.transformer_encoder = Encoder(
                                    dim = dim_model,
                                    depth = 1,
                                    heads = 8,
                                    attn_dropout = self.dropout,
                                    ff_dropout = self.dropout,
                                    attn_flash = True)
    
    self.cls_embedding = nn.Parameter(torch.zeros(1, 1, self.dim_model), requires_grad=True)

    print('Applying Xavier Uniform Init to x-transformer following torch.Transformer')
    self._apply_xavier_init()
    print('Adding dropout after feedforward layer in x-transformer')
    self._add_dropout_after_ff()
    print('Adding dropout after attention layer in x-transformer')
    self._add_dropout_after_attn()

  def _add_dropout_after_attn(self):
    for layer in self.transformer_encoder.layers:
      if 'Attention' in str(type(layer[1])): 
        if isinstance(layer[1].to_out, nn.Sequential): # if GLU
          layer[1].to_out.append(nn.Dropout(self.dropout))
        elif isinstance(layer[1].to_out, nn.Linear): # if simple linear
          layer[1].to_out = nn.Sequential(layer[1].to_out, nn.Dropout(self.dropout))
        else:
          raise ValueError('to_out should be either nn.Sequential or nn.Linear')

  def _add_dropout_after_ff(self):
    for layer in self.transformer_encoder.layers:
      if 'FeedForward' in str(type(layer[1])):
        layer[1].ff.append(nn.Dropout(self.dropout))

  def _apply_xavier_init(self):
    for name, param in self.transformer_encoder.named_parameters():
      if 'to_q' in name or 'to_k' in name or 'to_v' in name:
          torch.nn.init.xavier_uniform_(param, gain=0.5**0.5)

  def _apply_window_on_input_vec(self, embeddings):
    window_size = 1
    zero_vec = torch.zeros(embeddings.shape[0], window_size-1, embeddings.shape[2], embeddings.shape[3]).to(embeddings.device) # B x (window_size-1) x num_features x emb_size
    window_applied_input_vec = torch.cat([zero_vec, embeddings], dim=1) # B x (T+window_size-1) x num_features x emb_size
    window_applied_input_vec = window_applied_input_vec.unfold(1, window_size, 1) # B x T x window_size x emb_size x num_features
    window_applied_input_vec = window_applied_input_vec.transpose(3, 4) # B x T x window_size x num_features x emb_size
    window_applied_input_vec = window_applied_input_vec.reshape(embeddings.shape[0]*embeddings.shape[1], -1, embeddings.shape[3]) # (B*T) x (num_features*window_size) x emb_size
    return window_applied_input_vec

  def _apply_pos_enc(self, tgt):
    pos = torch.arange(tgt.shape[1]).to(tgt.device) # (num_features*window_size+1)
    pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1) # (B*T) x (num_features*window_size+1)
    tgt_pos = tgt + self.pos_enc(pos.long()) # (B*T) x (num_features*window_size+1) x emb_size
    return tgt_pos

  def forward(self, input_tokens):
    '''
    input_tokens: B x T x num_features
    '''
    # prepare input vector
    emb_list = [module(input_tokens[..., i]) for i, module in enumerate(self.layers)] # B x T x 1 x emb_size
    stacked_emb = torch.stack(emb_list, dim=2) # B x T x num_features x emb_size
    # apply window
    stacked_emb = self._apply_window_on_input_vec(stacked_emb)
    # add CLS
    cls = self.cls_embedding.repeat(stacked_emb.shape[0], 1, 1) # (B*T) x 1 x emb_size
    input_emb = torch.cat([stacked_emb, cls], dim=1) # (B*T) x (num_features*window_size+1) x emb_size
    output = self.transformer_encoder(input_emb) # (B*T) x (num_features*window_size+1) x emb_size
    # extract CLS
    output = output[:, -1, :].reshape((input_tokens.shape[0], input_tokens.shape[1], -1)) # B x T x emb_size
    return output

class RVQMultiEmbedding(nn.Module):
  def __init__(
    self, 
    vocab:LangTokenVocab, 
    dim_model:int
  ):
    super().__init__()
    self.vocab_size = vocab.get_vocab_size()
    self.dim_model = dim_model
    self.features = vocab.feature_list
    self.layers = []
    self._make_emb_layers()

  def _make_emb_layers(self):
    vocab_sizes = [self.vocab_size[key] for key in self.features]
    self.embedding_sizes = [self.dim_model for _ in self.features]
    for vocab_size, embedding_size in zip(vocab_sizes, self.embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    embeddings = torch.zeros(x.shape[0], x.shape[1], self.dim_model).to(x.device)
    emb_list = [module(x[:, (idx+1)%4::4]) for idx, module in enumerate(self.layers)]
    for idx, emb in enumerate(emb_list):
      embeddings[:, (idx+1)%4::4] = emb
    return embeddings
  
  def get_emb_by_key(self, key:str, token:torch.Tensor):
    layer_idx = self.features.index(key)
    return self.layers[layer_idx](token)

class XtransformerDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq)
      
class XtransformerCrossAttendDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True,
                                    cross_attend = True,
                                    only_cross = False)
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    assert context is not None or context_embedding is not None, 'context or context_embedding should be provided for prefix decoder'
    if context_embedding is None:
      input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
      attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
      assert input_ids is not None, 'input_ids should be provided for prefix decoder'
      assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
      assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'

      context = self.text_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask
      ).last_hidden_state
    else:
      context = context_embedding
    
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True, context=context)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, context=context, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq, context=context)
 
class XtransformerLargeCrossAttendDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-large')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True,
                                    cross_attend = True,
                                    only_cross = False)
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    assert context is not None or context_embedding is not None, 'context or context_embedding should be provided for prefix decoder'
    if context_embedding is None:
      input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
      attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
      assert input_ids is not None, 'input_ids should be provided for prefix decoder'
      assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
      assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'

      context = self.text_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask
      ).last_hidden_state
    else:
      context = context_embedding
    
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True, context=context)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, context=context, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq, context=context)
 
class NewCrossAttendDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True,
                                    cross_attend = True,
                                    only_cross = False,
                                    use_rmsnorm=True,
                                  ff_swish = True, # set this to True
                                  ff_glu = True,    # set to true to use for all feedforwards
                                  )  
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    assert context is not None or context_embedding is not None, 'context or context_embedding should be provided for prefix decoder'
    if context_embedding is None:
      input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
      attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
      assert input_ids is not None, 'input_ids should be provided for prefix decoder'
      assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
      assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'

      context = self.text_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask
      ).last_hidden_state
    else:
      context = context_embedding
    
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True, context=context)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, context=context, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq, context=context)
     
class NewCrossAttendwithRoPEDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True,
                                    cross_attend = True,
                                    only_cross = False,
                                    use_rmsnorm=True,
                                    rotary_pos_emb = True,
                                  ff_swish = True, # set this to True
                                  ff_glu = True,    # set to true to use for all feedforwards
                                  )  
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    assert context is not None or context_embedding is not None, 'context or context_embedding should be provided for prefix decoder'
    if context_embedding is None:
      input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
      attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
      assert input_ids is not None, 'input_ids should be provided for prefix decoder'
      assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
      assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'

      context = self.text_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask
      ).last_hidden_state
    else:
      context = context_embedding
    
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True, context=context)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, context=context, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq, context=context)

class XtransformerPrefixDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = PrefixDecoder(
                                    dim = dim,
                                    depth = depth,
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

  def forward(self, seq, cache=None,train=False,context=None):
    assert context is not None, 'context should be provided for prefix decoder'
    input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
    attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
    assert input_ids is not None, 'input_ids should be provided for prefix decoder'
    assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
    assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'
    context = self.text_encoder(
      input_ids=input_ids, 
      attention_mask=attention_mask
    ).last_hidden_state
    
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq)
      
class XtransformerPretrainingDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
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

  def forward(self, seq, cache=None,train=False,context=None, context_embedding=None):

    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True)
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, return_hiddens=True)
        return hidden_vec, intermediates
      else:
        return self.transformer_decoder(seq)
      
class XtransformerFinetuningDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    assert context is not None or context_embedding is not None, 'context or context_embedding should be provided for prefix decoder'
    if context_embedding is None:
      input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
      attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
      assert input_ids is not None, 'input_ids should be provided for prefix decoder'
      assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
      assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'

      context = self.text_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask,
      ).last_hidden_state
    else:
      context = context_embedding
    
    # concatenate context with seq
    seq = torch.cat([context, seq], dim=1) # B x (T+context_length) x emb_size
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True)
      # cut to only return the seq part
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, return_hiddens=True)
        # cut to only return the seq part
        hidden_vec = hidden_vec[:, context.shape[1]:, :]
        return hidden_vec, intermediates
      else:
        # cut to only return the seq part
        hidden_vec = self.transformer_decoder(seq)
        hidden_vec = hidden_vec[:, context.shape[1]:, :]
        return hidden_vec
      
class XtransformerLargeFinetuningDecoder(nn.Module):
  def __init__(
      self, 
      dim:int,
      depth:int,
      heads:int,
      dropout:float
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-large')
    # frozen text encoder
    for param in self.text_encoder.parameters():
      param.requires_grad = False
      
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = depth,
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

  def forward(self, seq, cache=None,train=False,context=None,context_embedding=None):
    assert context is not None or context_embedding is not None, 'context or context_embedding should be provided for prefix decoder'
    if context_embedding is None:
      input_ids = context['input_ids'].squeeze(1) if context['input_ids'].ndim == 3 else context['input_ids']
      attention_mask = context['attention_mask'].squeeze(1) if context['attention_mask'].ndim == 3 else context['attention_mask']
      assert input_ids is not None, 'input_ids should be provided for prefix decoder'
      assert attention_mask is not None, 'attention_mask should be provided for prefix decoder'
      assert input_ids.device == self.text_encoder.device, 'input_ids should be on the same device as text_encoder'

      context = self.text_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask,
      ).last_hidden_state
    else:
      context = context_embedding
    
    # concatenate context with seq
    seq = torch.cat([context, seq], dim=1) # B x (T+context_length) x emb_size
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, cache=cache, return_hiddens=True)
      # cut to only return the seq part
      return hidden_vec, intermediates
    else:
      if train:
        hidden_vec, intermediates = self.transformer_decoder(seq, return_hiddens=True)
        # cut to only return the seq part
        hidden_vec = hidden_vec[:, context.shape[1]:, :]
        return hidden_vec, intermediates
      else:
        # cut to only return the seq part
        hidden_vec = self.transformer_decoder(seq)
        hidden_vec = hidden_vec[:, context.shape[1]:, :]
        return hidden_vec