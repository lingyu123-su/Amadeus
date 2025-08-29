import re
import random
from pathlib import Path
from collections import OrderedDict
from typing import Union, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
# lock of thread
from threading import Lock

import json
from tqdm import tqdm
from torch.utils.data import Dataset,IterableDataset
from transformers import T5Tokenizer

from .augmentor import Augmentor
from .compile_utils import VanillaTransformer_compiler
from data_representation import vocab_utils

def get_emb_total_size(config, vocab):
  emb_param = config.nn_params.emb
  total_size = 0 
  for feature in vocab.feature_list:
    size = int(emb_param[feature] * emb_param.emb_size)
    total_size += size
    emb_param[feature] = size
  emb_param.total_size = total_size
  config.nn_params.emb = emb_param
  return config

class TuneCompiler(Dataset):
  def __init__(
      self, 
      data:List[Tuple[np.ndarray, str]], 
      data_type:str, 
      augmentor:Augmentor, 
      vocab:vocab_utils.LangTokenVocab,
      input_length:int,
      first_pred_feature:str,
      caption_path:Union[str, None] = None,
      for_evaluation: bool = False
  ):
    '''
    The data is distributed on-the-fly by the TuneCompiler
    Pitch, Chord augementation is applied to the training data every iteration
    Segmentation is applied every epoch for the training data
    '''
    super().__init__()
    self.data_list = data
    self.data_type = data_type
    self.augmentor = augmentor
    self.eos_token = vocab.eos_token
    self.compile_function = VanillaTransformer_compiler(
      data_list=self.data_list, 
      augmentor=self.augmentor, 
      eos_token=self.eos_token, 
      input_length=input_length,
      first_pred_feature=first_pred_feature,
      encoding_scheme=vocab.encoding_scheme
    )
    self.segment2tune_name = None
    self.tune_name2segment = None
    self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large",legacy=False)  # Initialize T5 tokenizer for caption processing


    if self.data_type == 'valid' or self.data_type == 'test':
      self._update_segments_for_validset()
    else:
      self._update_segments_for_trainset()

  def _update_segments_for_trainset(self, random_seed=0):
    random.seed(random_seed)
    if self.segment2tune_name is not None:
      # If segments are already compiled, we can skip the compilation
      print("Segments are already compiled, skipping compilation")
      return
    print("Compiling segments for training data")
    with Lock():
      self.segments, _, self.segment2tune_name = self.compile_function.make_segments(self.data_type)
    print(f"number of trainset segments: {len(self.segments)}")

  def _update_segments_for_validset(self, random_seed=0):
    random.seed(random_seed)
    with Lock():
      self.segments, self.tune_name2segment, self.segment2tune_name = self.compile_function.make_segments(self.data_type)
    print(f"number of testset segments: {len(self.segments)}")

  def __getitem__(self, idx):
    segment, tensor_mask = self.segments[idx]
    tune_name = self.segment2tune_name[idx]
    try:
      encoded_caption = self.t5_tokenizer(tune_name, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    except Exception as e:
      print(f"Error encoding caption for tune {tune_name}: {e}")
      encoded_caption = self.t5_tokenizer("No caption available", return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    return segment, tensor_mask, tune_name, encoded_caption
    if self.data_type == 'train':
      augmented_segment = self.augmentor(segment)
      return augmented_segment, tensor_mask, tune_name, encoded_caption
    else:
      return segment, tensor_mask, tune_name, encoded_caption
  
  def get_segments_with_tune_idx(self, tune_name, seg_order):
    '''
    This function is used to retrieve the segment with the tune name and segment order during the validation
    '''
    segments_list = self.tune_name2segment[tune_name]
    segment_idx = segments_list[seg_order]
    segment, mask = self.segments[segment_idx][0], self.segments[segment_idx][1]
    return segment, mask

  def __len__(self):
    return len(self.segments)

class IterTuneCompiler(IterableDataset):
  def __init__(
      self, 
      data: List[Tuple[np.ndarray, str]], 
      data_type: str, 
      augmentor: Augmentor, 
      vocab: vocab_utils.LangTokenVocab,
      input_length: int,
      first_pred_feature: str,
      caption_path: Union[str, None] = None,
      for_evaluation: bool = False
  ):
    '''
    The data is distributed on-the-fly by the IterTuneCompiler.
    Pitch, Chord augmentation is applied to the training data every iteration.
    Segmentation is applied every epoch for the training data.
    '''
    super().__init__()
    self.data_list = data
    self.data_type = data_type
    self.augmentor = augmentor
    self.eos_token = vocab.eos_token
    self.compile_function = VanillaTransformer_compiler(
      data_list=self.data_list, 
      augmentor=self.augmentor, 
      eos_token=self.eos_token, 
      input_length=input_length,
      first_pred_feature=first_pred_feature,
      encoding_scheme=vocab.encoding_scheme
    )
    self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
    self.random_seed = 0

  def __iter__(self):
    # This will yield ([segment, mask], tune_name2segment, segment2tune_name)
    generator = self.compile_function.make_segments_iters(self.data_type)
    for ([segment, mask], tune_name2segment, segment2tune_name) in generator:
      # print(len(segment2tune_name), len(tune_name2segment))
      tune_name = segment2tune_name[-1]  # Get the last tune name from the segment2tune_name list
      # print(f"Processing tune: {tune_name}")
      try:
        encoded_caption = self.t5_tokenizer(tune_name, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
      except Exception as e:
        encoded_caption = self.t5_tokenizer("No caption available", return_tensors='pt', padding='max_length', truncation=True, max_length=128)
      if self.data_type == 'train':
        segment = self.augmentor(segment)
      # use input_ids replace tune_name
      tune_name = encoded_caption['input_ids'][0]  # Use the input_ids from the encoded caption
      yield segment, mask, tune_name, encoded_caption

  def __len__(self):
    # If you want to use __len__, you need to know the number of segments in advance.
    # Otherwise, you can raise an exception or return a default value.
    raise NotImplementedError("IterTuneCompiler is an iterable dataset and does not support __len__.")
  
class SymbolicMusicDataset(Dataset):
  def __init__(
      self, 
      vocab: vocab_utils.LangTokenVocab,
      encoding_scheme: str,                
      num_features: int,                   
      debug: bool,                         
      aug_type: Union[str, None],          
      input_length: int,                   
      first_pred_feature: str,
      caption_path: Union[str, None] = None,
      for_evaluation: bool = False           
  ):
    '''
    The vocabulary containing token representations for the dataset
    The encoding scheme used for representing symbolic music (e.g., REMI, NB, etc.)
    The number of features used for the dataset
    Debug mode; limits dataset size for faster testing if enabled
    Type of data augmentation to apply, if 'random' the compiler will apply pitch and chord augmentation
    Length of the input sequence for each sample
    Feature to predict first which is used for compound shift for NB, if not shift, 'type' is used
    '''
    super().__init__()
    # Initializing instance variables
    self.encoding_scheme = encoding_scheme
    self.num_features = num_features
    self.debug = debug
    self.input_length = input_length
    self.first_pred_feature = first_pred_feature
    self.caption_path = caption_path
    self.for_evaluation = for_evaluation

    # Load the vocabulary passed into the constructor
    self.vocab = vocab
    
    # Initialize augmentor for data augmentation
    self.augmentor = Augmentor(vocab=self.vocab, aug_type=aug_type, input_length=input_length)
    
    # Load preprocessed tune indices
    if self.for_evaluation:
      # For evaluation, we load the tune indices without any augmentation
      self.tune_in_idx, self.len_tunes, self.file_name_list = [], [], []
    else:
      self.tune_in_idx, self.len_tunes, self.file_name_list = self._load_tune_in_idx()
    # Plot the histogram of tune lengths for analysis
    dataset_name = self.__class__.__name__  # Get the class name (dataset name)
    len_dir_path = Path(f"len_tunes/{dataset_name}")  # Directory to store tune length histograms
    len_dir_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    if self. for_evaluation is False:
      self._plot_hist(self.len_tunes, len_dir_path / f"len_{encoding_scheme}{num_features}.png")

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    # Load preprocessed tune indices from .npz files
    print("preprocessed tune_in_idx data is being loaded")
    
    # List of files containing tune index data
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    
    # If debug mode is enabled, limit the number of loaded files
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]

    # Initialize dictionaries and lists for storing tune index data, tune lengths, and file names
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # Load tune index data from each .npz file
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']  # Load the numpy array from the file
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx  # Store the tune indices in the dictionary
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)  # Record the length of the tune
      file_name_list.append(tune_in_idx_file.stem)  # Append the file name (without extension)
    
    return tune_in_idx_dict, len_tunes, file_name_list  # Return the data structures

  def _plot_hist(self, len_tunes, path_outfile):
    # Plot histogram of tune lengths and save the plot
    Path(path_outfile).parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory for the plot exists
    
    # Convert tune lengths to a NumPy array
    data = np.array(list(len_tunes.values()))
    
    # Compute mean and standard deviation of tune lengths
    self.mean_len_tunes = np.mean(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    # cumpute the total length of all tunes
    self.total_len_tunes = np.sum(data)
    
    # Plot the histogram
    plt.figure(dpi=100)
    plt.hist(data, bins=50)
    plt.title(f"mean: {data_mean:.2f}, std: {data_std:.2f}, total: {self.total_len_tunes}, num_tunes: {len(data)}")
    plt.savefig(path_outfile)  # Save the plot to file
    plt.close()  # Close the plot to free memory

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    # Split the dataset into train, validation, and test sets based on the given ratio
    shuffled_tune_names = list(self.tune_in_idx.keys())  # Get the list of all tune names
    random.seed(seed)  # Set the seed for reproducibility
    random.shuffle(shuffled_tune_names)  # Shuffle the tune names
    
    # Compute the number of training, validation, and test samples
    num_train = int(len(shuffled_tune_names) * ratio)
    num_valid = int(len(shuffled_tune_names) * (1 - ratio) / 2)
    
    # Split the tune names into training, validation, and test sets
    train_names = shuffled_tune_names[:num_train]
    valid_names = shuffled_tune_names[num_train:num_train + num_valid]
    test_names = shuffled_tune_names[num_train + num_valid:]
    
    return train_names, valid_names, test_names, shuffled_tune_names  # Return the split lists

  def split_train_valid_test_set(self, dataset_name=None, ratio=None, seed=42, save_dir=None, for_evaluation: bool = False):
    # Split the dataset into train, validation, and test sets or load an existing split
    if not Path(f"metadata/{dataset_name}_caption_metadata.json").exists():
      # If no metadata exists, perform a random split and save metadata
      assert ratio is not None, "ratio should be given when you make metadata for split"
      
      # Perform the split
      train_names, valid_names, test_names, shuffled_tune_names = self._get_split_list_from_tune_in_idx(ratio, seed)
      
      # Log the split information
      print(f"Randomly split train and test set using seed {seed}")
      out_dict = {'shuffle_seed': seed,  # Seed used for shuffling
                  'shuffled_names': shuffled_tune_names,  # Shuffled list of tune names
                  'train': train_names,  # Training set names
                  'valid': valid_names,  # Validation set names
                  'test': test_names}  # Test set names
      
      # Save the split metadata to a JSON file
      with open(f"metadata/{dataset_name}_caption_metadata.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    else:
      # If metadata already exists, load it
      with open(f"metadata/{dataset_name}_caption_metadata.json", "r") as f:
        out_dict = json.load(f)
      
      # Ensure that the loaded data matches the current dataset
      train_names, valid_names, test_names = out_dict['train'], out_dict['valid'], out_dict['test']
      if self.for_evaluation is False:
        assert set(out_dict['shuffled_names']) == set(self.tune_in_idx.keys()), "Loaded data is not matched with the recorded metadata"

    # Prepare training, validation, and test datasets using the TuneCompiler
    if self.for_evaluation:
      # For evaluation, we do not need to create train and valid datasets
      train_data = []
      valid_data = []
      self.test_data = []
    else:
      train_data = [(self.tune_in_idx[tune_name], tune_name) for tune_name in train_names]
      valid_data = [(self.tune_in_idx[tune_name], tune_name) for tune_name in valid_names]
      self.test_data = [(self.tune_in_idx[tune_name], tune_name) for tune_name in test_names]

    # Initialize TuneCompiler objects for each split
    # if self.for_evaluation:
    #   train_dataset = None  # No training dataset for evaluation
    #   valid_dataset = None
    #   test_dataset = TuneCompiler(data=self.test_data, data_type='test', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)
    # else:
    train_dataset = IterTuneCompiler(data=train_data, data_type='train', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)
    valid_dataset = TuneCompiler(data=valid_data, data_type='valid', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)
    test_dataset = TuneCompiler(data=self.test_data, data_type='test', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)

    # Save metadata to a directory if specified
    if save_dir is not None:
      Path(save_dir).mkdir(parents=True, exist_ok=True)
      with open(Path(save_dir) / f"{dataset_name}_metadata.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    
    # Return the datasets for training, validation, and testing
    return train_dataset, valid_dataset, test_dataset

class Pop1k7(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)
    

class SymphonyMIDI(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

class LakhClean(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if tune_in_idx_file.stem in irregular_tunes:
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class LakhClean(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if tune_in_idx_file.stem in irregular_tunes:
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class ariamidi(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if tune_in_idx_file.stem in irregular_tunes:
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class gigamidi(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if tune_in_idx_file.stem in irregular_tunes:
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class PretrainingDataset(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)
  
  def _load_tune_in_idx_aria(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Load preprocessed tune indices for the aria dataset
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_ariamidi/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  def _load_tune_in_idx_giga(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Load preprocessed tune indices for the gigamidi dataset
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_gigamidi/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if "drums-only" in tune_in_idx_file.stem:
        print(f"skipping {tune_in_idx_file.stem} as it is a drums-only file")
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  def _load_tune_in_idx_pop1k7(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Load preprocessed tune indices for the Pop1k7 dataset
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_pop1k7/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  def _load_tune_in_idx_sod(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Load preprocessed tune indices for the SOD dataset
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_SOD/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  def _load_tune_in_idx_lakh(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''

    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_LakhALLFined/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.lakh_caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem.replace("_", "/", 1) + ".mid"
      location_key = f"lmd_full/{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue

      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  
  def _load_tune_in_idx_xmidi(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_XMIDI_Dataset/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.xmidi_caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem + ".midi"
      location_key = f"{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  
  def _load_tune_in_idx_new(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_new_dataset/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.new_caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem + ".mid"
      location_key = f"new_data_new_dataset/{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    self.lakh_caption_path =  "dataset/represented_data/tuneidx/train_set.json"
    self.xmidi_caption_path = "dataset/represented_data/tuneidx/all_captions.json"
    self.new_caption_path = "dataset/represented_data/tuneidx/new_dataset_captions_final.jsonl"

    # load all tune_in_idx data from aria, giga datasets
    tune_in_idx_giga, len_tunes_giga, file_name_list_giga = self._load_tune_in_idx_giga()
    tune_in_idx_aria, len_tunes_aria, file_name_list_aria = self._load_tune_in_idx_aria()
    tune_in_idx_lakh, len_tunes_lakh, file_name_list_lakh = self._load_tune_in_idx_lakh()
    tune_in_idx_xmidi, len_tunes_xmidi, file_name_list_xmidi = self._load_tune_in_idx_xmidi()
    tune_in_idx_new, len_tunes_new, file_name_list_new = self._load_tune_in_idx_new()

    # merge the two datasets
    tune_in_idx = {**tune_in_idx_aria, **tune_in_idx_giga, **tune_in_idx_lakh, **tune_in_idx_xmidi, **tune_in_idx_new}
    len_tunes = {**len_tunes_aria, **len_tunes_giga, **len_tunes_lakh, **len_tunes_xmidi, **len_tunes_new}
    file_name_list = file_name_list_aria + file_name_list_giga + file_name_list_lakh + file_name_list_xmidi + file_name_list_new
    print(f"number of loaded tunes: {len(tune_in_idx)}")
    return tune_in_idx, len_tunes, file_name_list
    
  
class SOD(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)
  
  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    with open("metadata/SOD_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if tune_in_idx_file.stem in irregular_tunes:
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

class BachChorale(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

class Pop909(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Pop909 dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"-v\d+$", "", tune) for tune in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names
  
  
class LakhALL(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        #   # remove file in tune_in_idx_list
        #   location2caption[item["location"]] = "test_set"
          # continue
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem.replace("_", "/", 1) + ".mid"
      location_key = f"lmd_full/{location_key}"
      try:
        caption = location2caption.get(location_key, None)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        print(f"Caption for {location_key} is None, skipping this tune")
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class LakhALLFined(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        # if item["test_set"] is True:
        #   continue # skip test set tunes
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem.replace("_", "/", 1) + ".mid"
      location_key = f"lmd_full/{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    # filter out none in tune_in_idx
    print("length of tune_in_idx before filtering:", len(self.tune_in_idx))
    try:
      self.tune_in_idx = {k: v for k, v in self.tune_in_idx.items() if v is not None}
    except:
      print("Error filtering None values in tune_in_idx, skipping filtering")
      return [], [], [], []
    print("length of tune_in_idx after filtering:", len(self.tune_in_idx))
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class XMIDI_Dataset(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        # if item["test_set"] is True:
        #   continue # skip test set tunes
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem + ".midi"
      print(f"Processing file: {tune_in_idx_file.stem}, location_key: {location_key}")
      location_key = f"{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    # filter out none in tune_in_idx
    print("length of tune_in_idx before filtering:", len(self.tune_in_idx))
    self.tune_in_idx = {k: v for k, v in self.tune_in_idx.items() if v is not None}
    print("length of tune_in_idx after filtering:", len(self.tune_in_idx))
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class new_dataset(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        # if item["test_set"] is True:
        #   continue # skip test set tunes
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem.split("/")[-1] + ".mid"
      location_key = f"new_data_new_dataset/{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    # filter out none in tune_in_idx
    print("length of tune_in_idx before filtering:", len(self.tune_in_idx))
    self.tune_in_idx = {k: v for k, v in self.tune_in_idx.items() if v is not None}
    print("length of tune_in_idx after filtering:", len(self.tune_in_idx))
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class SymphonyNet_Dataset(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path)

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem.replace("_", "/", 1) + ".mid"
      location_key = f"/data2/suhongju/research/music-generation/BandZero/SymphonyNet_Dataset/{location_key}"
      try:
        caption = location2caption.get(location_key, None)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        print(f"Caption for {location_key} is None, skipping this tune")
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = shuffled_tune_names
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

# use lakhAllFined, XMIDI_dataset, new_dataset,  as finetune dataset
class FinetuneDataset(SymbolicMusicDataset):
  def __init__(self, vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path=None,
                for_evaluation: bool = False):
    super().__init__(vocab, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature, caption_path,
                     for_evaluation=for_evaluation)
    
    
  def _load_tune_in_idx_lakh(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''

    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_LakhALLFined/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.lakh_caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem.replace("_", "/", 1) + ".mid"
      location_key = f"lmd_full/{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue

      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  
  def _load_tune_in_idx_xmidi(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_XMIDI_Dataset/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.xmidi_caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem + ".midi"
      location_key = f"{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  
  def _load_tune_in_idx_new(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Irregular tunes are removed from the dataset for better generation quality
    It includes tunes that are not quantized properly, mostly theay are expressive performance data
    '''
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_new_dataset/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    
    # load caption
    self.caption_list = []
    with open(self.new_caption_path, "r") as f:
      # every line is a caption for the tune
      for line in f:
        self.caption_list.append(line.strip())
    print(f"number of loaded captions: {len(self.caption_list)}")

    with open("metadata/LakhClean_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    
    # 构建 location 到 caption 的映射
    location2caption = {}
    for line in self.caption_list:
      try:
        # 假设每行是一个json字符串
        item = json.loads(line)
        location2caption[item["location"]] = item["caption"]
      except Exception:
        continue

    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      # 0_06d3f5a5954848ba13b9128f68f0a1d1 -> 0/06d3f5a5954848ba13b9128f68f0a1d1
      location_key = tune_in_idx_file.stem + ".mid"
      location_key = f"new_data_new_dataset/{location_key}"
      try:
        caption = location2caption.get(location_key)
      except KeyError:
        print(f"KeyError: {location_key} not found in location2caption")
        continue
      if caption is None:
        continue
      # print(tune_in_idx_file.stem, location_key, caption)
      # print("*" * 20)
      # 你可以在这里使用caption变量
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[caption] = tune_in_idx
      len_tunes[caption] = len(tune_in_idx)
      file_name_list.append(caption)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list
  
  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    '''
    Load tune_in_idx from all three datasets
    '''
    self.lakh_caption_path =  "dataset/represented_data/tuneidx/train_set.json"
    self.xmidi_caption_path = "dataset/represented_data/tuneidx/all_captions.json"
    self.new_caption_path = "dataset/represented_data/tuneidx/new_dataset_captions_final.jsonl"

    tune_in_idx_lakh, len_tunes_lakh, file_name_list_lakh = self._load_tune_in_idx_lakh()
    tune_in_idx_xmidi, len_tunes_xmidi, file_name_list_xmidi = self._load_tune_in_idx_xmidi()
    tune_in_idx_new, len_tunes_new, file_name_list_new = self._load_tune_in_idx_new()
    # 合并三个数据集
    tune_in_idx = {**tune_in_idx_lakh, **tune_in_idx_xmidi, **tune_in_idx_new}
    len_tunes = {**len_tunes_lakh, **len_tunes_xmidi, **len_tunes_new}
    file_name_list = file_name_list_lakh + file_name_list_xmidi + file_name_list_new
    print(f"number of loaded tunes: {len(tune_in_idx)}")
    return tune_in_idx, len_tunes, file_name_list 
  
  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    '''
    As Lakh dataset contains multiple versions of the same song, we split the dataset based on the song name
    '''
    # filter out none in tune_in_idx
    print("length of tune_in_idx before filtering:", len(self.tune_in_idx))
    try:
      self.tune_in_idx = {k: v for k, v in self.tune_in_idx.items() if v is not None}
    except:
      print("Error filtering None values in tune_in_idx, skipping filtering")
      return [], [], [], []
    print("length of tune_in_idx after filtering:", len(self.tune_in_idx))
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names
    