import torch
from pathlib import Path
import argparse
import json
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig
from transformers import T5Tokenizer, T5EncoderModel
from Amadeus.train_utils import adjust_prediction_order

from Amadeus.evaluation_utils import (
    get_dir_from_wandb_by_code,
    wandb_style_config_to_omega_config,
)
from Amadeus.symbolic_encoding import decoding_utils, data_utils
from data_representation import vocab_utils
from Amadeus import  model_zoo
from Amadeus.symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-wandb_exp_dir",
        required=True,
        type=str,
        help="wandb experiment directory",
    )
    parser.add_argument(
        "-prompt",
        required=True,
        type=str,
        help="text prompt for generation",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="outputs",
        help="directory to save results",
    )
    parser.add_argument(
        "-sampling_method",
        type=str,
        choices=('top_p', 'top_k'),
        default='top_p',
        help="sampling method",
    )
    parser.add_argument(
        "-threshold",
        type=float,
        default=0.99,
        help="threshold",
    )
    parser.add_argument(
        "-temperature",
        type=float,
        default=1.15,
        help="temperature",
    )
    parser.add_argument(
        "-generate_length",
        type=int,
        default=2048,
        help="length of the generated sequence",
    )
    parser.add_argument(
        "-text_encoder_model",
        type=str,
        default='google/flan-t5-large',
        help="pretrained text encoder model",
    )
    return parser

def get_best_ckpt_path_and_config(dir):
  if dir is None:
    raise ValueError('No such code in wandb_dir')
  ckpt_dir = dir / 'files' / 'checkpoints'

  config_path = dir / 'files'  / 'config.yaml'
  # print all files in ckpt_dir
  vocab_path = next(ckpt_dir.glob('vocab*'))

  # if there is pt file ending with 'last', return it 
  if len(list(ckpt_dir.glob('*last.pt'))) > 0:
    last_ckpt_fn = next(ckpt_dir.glob('*last.pt'))
  else:
    pt_fns = sorted(list(ckpt_dir.glob('*.pt')), key=lambda fn: int(fn.stem.split('_')[0].replace('iter', '')))
    last_ckpt_fn = pt_fns[-1]

  return last_ckpt_fn, config_path, vocab_path

def prepare_model_and_dataset_from_config(config: DictConfig, vocab_path:str):
    nn_params = config.nn_params
    vocab_path = Path(vocab_path)

    # print(config)
    encoding_scheme = config.nn_params.encoding_scheme
    num_features = config.nn_params.num_features
    
    # get vocab
    vocab_name = {'remi':'LangTokenVocab', 'cp':'MusicTokenVocabCP', 'nb':'MusicTokenVocabNB'}
    selected_vocab_name = vocab_name[encoding_scheme]

    vocab = getattr(vocab_utils, selected_vocab_name)(
      in_vocab_file_path=vocab_path,
      event_data=None,
      encoding_scheme=encoding_scheme, 
      num_features=num_features)
        # get proper prediction order according to the encoding scheme and target feature in the config
    prediction_order = adjust_prediction_order(encoding_scheme, num_features, config.data_params.first_pred_feature, nn_params)

    # Create the Transformer model based on configuration parameters
    AmadeusModel = getattr(model_zoo, nn_params.model_name)(
                          vocab=vocab,
                          input_length=config.train_params.input_length,
                          prediction_order=prediction_order,
                          input_embedder_name=nn_params.input_embedder_name,
                          main_decoder_name=nn_params.main_decoder_name,
                          sub_decoder_name=nn_params.sub_decoder_name,
                          sub_decoder_depth=nn_params.sub_decoder.num_layer if hasattr(nn_params, 'sub_decoder') else 0,
                          sub_decoder_enricher_use=nn_params.sub_decoder.feature_enricher_use \
                            if hasattr(nn_params, 'sub_decoder') and hasattr(nn_params.sub_decoder, 'feature_enricher_use') else False,
                          dim=nn_params.main_decoder.dim_model,
                          heads=nn_params.main_decoder.num_head,
                          depth=nn_params.main_decoder.num_layer,
                          dropout=nn_params.model_dropout,
                          )
    
    return AmadeusModel, [], vocab

def load_resources(dir, device):
    """Load model and dataset resources"""
    dir = Path(dir)
    ckpt_path, config_path, vocab_path = get_best_ckpt_path_and_config(
        dir
    )
    config = OmegaConf.load(config_path)
    config = wandb_style_config_to_omega_config(config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model, _, vocab = prepare_model_and_dataset_from_config(config, vocab_path)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    torch.compile(model)
    print("total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return config, model, vocab

def generate_with_text_prompt(config, vocab, model, device, prompt, save_dir,
                              first_pred_feature, sampling_method, threshold,
                              temperature, generation_length=1024):
    encoding_scheme = config.nn_params.encoding_scheme
    tokenizer = T5Tokenizer.from_pretrained(config.text_encoder_model)
    encoder = T5EncoderModel.from_pretrained(config.text_encoder_model).to(device)
    print(f"Using T5EncoderModel for text prompt:\n{prompt}")
    context = tokenizer(prompt, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=128).to(device)
    context = encoder(**context).last_hidden_state

    in_beat_resolution_dict = {'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4}
    in_beat_resolution = in_beat_resolution_dict.get(config.dataset, 4)

    midi_decoder_dict = {'remi': 'MidiDecoder4REMI',
                         'cp': 'MidiDecoder4CP',
                         'nb': 'MidiDecoder4NB'}
    decoder_name = midi_decoder_dict[encoding_scheme]
    decoder = getattr(decoding_utils, decoder_name)(
        vocab=vocab, in_beat_resolution=in_beat_resolution, dataset_name=config.dataset
    )

    generated_sample = model.generate(
        0, generation_length, condition=None, num_target_measures=None,
        sampling_method=sampling_method, threshold=threshold,
        temperature=temperature, context=context
    )
    if encoding_scheme == 'nb':
        generated_sample = reverse_shift_and_pad_for_tensor(generated_sample, first_pred_feature)

    save_dir.mkdir(parents=True, exist_ok=True)

    output_file = save_dir / f"generated.mid"
    decoder(generated_sample, output_path=str(output_file))
    print(f"Generated file saved at: {output_file}")


def main():
    args = get_argument_parser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, model, vocab = load_resources(args.wandb_exp_dir, device)

    save_dir = Path(args.output_dir)
    config.text_encoder_model = args.text_encoder_model
    generate_with_text_prompt(
        config,
        vocab,
        model,
        device,
        args.prompt,
        save_dir,
        config.data_params.first_pred_feature,
        args.sampling_method,
        args.threshold,
        args.temperature,
        generation_length=args.generate_length,
    )


if __name__ == "__main__":
    main()