import torch
import torch.nn.functional as F

def top_p_sampling(logits, thres=0.9):
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

  sorted_indices_to_remove = cum_probs > thres
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0

  # Create an empty tensor to hold the new logits
  new_logits = logits.clone()

  # Use the sorted indices to place the '-inf' in the original places
  indices_to_remove = sorted_indices[sorted_indices_to_remove]
  new_logits[..., indices_to_remove] = float('-inf')
  return new_logits


# refered: https://github.com/cimeister/typical-sampling
def typical_sampling(logits, thres=0.99):
  # calculate entropy
  normalized = torch.nn.functional.log_softmax(logits, dim=-1)
  p = torch.exp(normalized)
  ent = -(normalized * p).nansum(-1, keepdim=True)

  # shift and sort
  shifted_scores = torch.abs((-normalized) - ent)
  sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
  sorted_logits = logits.gather(-1, sorted_indices)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

  # Remove tokens with cumulative mass above the threshold
  last_ind = (cumulative_probs < thres).sum(dim=-1)
  last_ind[last_ind < 0] = 0
  sorted_indices_to_remove = sorted_scores > sorted_scores.gather(-1, last_ind.view(-1, 1, 1))
  # if self.min_tokens_to_keep > 1:
  #     # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
  #     sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
  indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)

  scores = logits.masked_fill(indices_to_remove, float("-inf"))
  return scores

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
  # 
# refered: https://github.com/john-hewitt/truncation-sampling
def eta_sampling(logits, epsilon) -> torch.FloatTensor:
  probabilities = logits.softmax(dim=-1)
  entropy = torch.distributions.Categorical(probs=probabilities).entropy()
  new_epsilon = min(epsilon, torch.sqrt(torch.tensor(epsilon))*torch.exp(-entropy))
  indices_to_remove = probabilities < new_epsilon
  max_word = torch.argmax(logits, dim=-1)
  indices_to_remove[..., max_word.squeeze()] = 0
  new_scores = logits.masked_fill(indices_to_remove, float("-inf"))
  return new_scores

def sample(logits, sampling_method, threshold, temperature):
  """Sample from the logits with a specific sampling strategy."""
  if sampling_method == "top_p":
    probs = F.softmax(top_p_sampling(logits, thres=threshold) / temperature, dim=-1)
  elif sampling_method == "typical":
    probs = F.softmax(typical_sampling(logits, thres=threshold) / temperature, dim=-1)
  elif sampling_method == "eta":
    probs = F.softmax(eta_sampling(logits, epsilon=threshold) / temperature, dim=-1)
  else:
    probs = F.softmax(logits / temperature, dim=-1)
  return torch.multinomial(probs[-1,-1,:], 1)

def sample_with_prob(logits, sampling_method, threshold, temperature):
    """Sample from the logits with a specific sampling strategy and return the token and its probability."""
    #  temporarily apply the sampling method to logits
    logits = logits / temperature
    # logits = add_gumbel_noise(logits, temperature)
  
    if sampling_method == "top_p":
        modified_logits = top_p_sampling(logits, thres=threshold)
    elif sampling_method == "typical":
        modified_logits = typical_sampling(logits, thres=threshold)
    elif sampling_method == "eta":
        modified_logits = eta_sampling(logits, epsilon=threshold)
    else:
        modified_logits = logits  # 其他情况直接使用原始logits
    
    # print(modified_logits.shape)
    # 应用温度调整并计算概率
    # probs = F.softmax(modified_logits / temperature, dim=-1)
    probs = F.softmax(modified_logits, dim=-1)
    
    # 获取最后一个位置的概率分布
    # probs_last = probs[-1, -1, :]
    # print(probs.shape)
    probs_last = probs[-1, -1, :]
    
    # 采样
    sampled_token = torch.multinomial(probs_last, num_samples=1)
    # 获取对应的概率值
    prob_value = probs_last[sampled_token]
    
    return sampled_token, prob_value.squeeze()

def top_p_sampling_fast(logits, thres=0.9):
    """
    logits: Tensor of shape [B, L, V]
    Returns: logits with low-prob tokens masked as -inf, shape [B, L, V]
    """
    # Step 1: sort logits and get indices
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)  # [B, L, V]
    
    # Step 2: compute cumulative probs
    probs = F.softmax(sorted_logits, dim=-1)  # [B, L, V]
    cum_probs = torch.cumsum(probs, dim=-1)   # [B, L, V]

    # Step 3: mask tokens beyond cumulative threshold
    sorted_mask = cum_probs > thres
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False  # always keep at least one token

    # Step 4: scatter back to original order
    # Create mask of same shape as logits, default False
    mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, L, V]
    mask = mask.scatter(-1, sorted_indices, sorted_mask)

    # Step 5: mask logits
    logits = logits.masked_fill(mask, float('-inf'))  # final masked logits

    return logits

def sample_with_prob_fast(logits, sampling_method="top_p", threshold=0.9, temperature=1.0, mask_indices=None):
    """
    logits: [B*T, num_sub_tokens, vocab_size]
    mask_indices: mask indicating which tokens to sample, shape = [B*T, num_sub_tokens]
    """
    if temperature != 1.0:
        logits = logits / temperature

    if sampling_method == "top_p":
        logits = top_p_sampling_fast(logits, thres=threshold)  # should support batch
    elif sampling_method == "typical":
        logits = typical_sampling(logits, thres=threshold)
    elif sampling_method == "eta":
        logits = eta_sampling(logits, epsilon=threshold)
    # else: keep logits as-is

    probs = torch.softmax(logits, dim=-1)  # [B*T, num_sub_tokens, vocab_size]

    B, L, V = probs.shape
    probs_flat = probs.view(-1, V)  # [(B*T * num_sub_tokens), V]

    # 采样：multinomial 不能一次性处理 3D，展平后采样
    sampled = torch.multinomial(probs_flat, num_samples=1)  # [(B*T * num_sub_tokens), 1]
    sampled = sampled.view(B, L)  # [B*T, num_sub_tokens]

    sampled_probs = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)  # [B*T, num_sub_tokens]

    return sampled, sampled_probs
