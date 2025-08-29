from email.mime import audio
import torch
from pathlib import Path
import json
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig
from transformers import T5Tokenizer, T5EncoderModel
import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Amadeus.train_utils import adjust_prediction_order
from Amadeus.evaluation_utils import (
    wandb_style_config_to_omega_config,
)
from Amadeus.symbolic_encoding import decoding_utils
from data_representation import vocab_utils
from Amadeus import model_zoo
from Amadeus.symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor


# === Keep original utility functions ===
def get_best_ckpt_path_and_config(dir):
    if dir is None:
        raise ValueError('No such code in wandb_dir')
    ckpt_dir = dir / 'files' / 'checkpoints'

    config_path = dir / 'files' / 'config.yaml'
    vocab_path = next(ckpt_dir.glob('vocab*'))

    if len(list(ckpt_dir.glob('*last.pt'))) > 0:
        last_ckpt_fn = next(ckpt_dir.glob('*last.pt'))
    else:
        pt_fns = sorted(list(ckpt_dir.glob('*.pt')), key=lambda fn: int(fn.stem.split('_')[0].replace('iter', '')))
        last_ckpt_fn = pt_fns[-1]

    return last_ckpt_fn, config_path, vocab_path


def prepare_model_and_dataset_from_config(config: DictConfig, vocab_path: str):
    nn_params = config.nn_params
    vocab_path = Path(vocab_path)

    encoding_scheme = config.nn_params.encoding_scheme
    num_features = config.nn_params.num_features
    vocab_name = {'remi': 'LangTokenVocab', 'cp': 'MusicTokenVocabCP', 'nb': 'MusicTokenVocabNB'}
    selected_vocab_name = vocab_name[encoding_scheme]

    vocab = getattr(vocab_utils, selected_vocab_name)(
        in_vocab_file_path=vocab_path,
        event_data=None,
        encoding_scheme=encoding_scheme,
        num_features=num_features)

    prediction_order = adjust_prediction_order(encoding_scheme, num_features, config.data_params.first_pred_feature, nn_params)

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
    return AmadeusModel, vocab


def load_resources(wandb_exp_dir, device):
    wandb_exp_dir = Path(wandb_exp_dir)
    ckpt_path, config_path, vocab_path = get_best_ckpt_path_and_config(
        wandb_exp_dir
    )
    config = OmegaConf.load(config_path)
    config = wandb_style_config_to_omega_config(config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model, vocab = prepare_model_and_dataset_from_config(config, vocab_path)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    torch.compile(model)
    print("total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return config, model, vocab


import time

def generate_with_text_prompt(config, vocab, model, device, prompt, text_encoder_model,
                              sampling_method='top_p', threshold=0.99,
                              temperature=1.15, generation_length=1024):
    encoding_scheme = config.nn_params.encoding_scheme
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_model)
    encoder = T5EncoderModel.from_pretrained(text_encoder_model).to(device)
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
        generated_sample = reverse_shift_and_pad_for_tensor(generated_sample, config.data_params.first_pred_feature)

    # === Generate filename with timestamp ===
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    Path("outputs").mkdir(exist_ok=True)
    output_file = Path("outputs") / f"generated_{timestamp}.mid"

    decoder(generated_sample, output_path=str(output_file))
    return str(output_file)

# === Gradio Demo ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = "models/Amadeus-S"  # 模型路径，可以是 Amadeus-S, Amadeus-M, Amadeus-L
# check if model exists
if not Path(model_id).exists():
    # download from huggingface
    import os
    from huggingface_hub import snapshot_download

    os.makedirs("models", exist_ok=True)

    local_dir = snapshot_download(
        repo_id="longyu1315/Amadeus-S",
        repo_type="model",
        local_dir="models"
    )

    print("模型已下载到:", local_dir)   
config, model, vocab = load_resources(model_id, device)

# Example prompts
examples = {
    "prompt1": "A melodic electronic ambient song with a touch of darkness, set in the key of E major and a 4/4 time signature. Tubular bells, electric guitar, synth effects, synth pad, and oboe weave together to create an epic, space-like atmosphere. The tempo is a steady Andante, and the chord progression of A, B, and E forms the harmonic backbone of this captivating piece.",
    "prompt2": "A melodic electronic song with a moderate tempo, featuring a blend of drums, piano, brass section, alto saxophone, and synth bass. The piece is set in B minor and follows a chord progression of C#m, B, A, and B. With a duration of 252 seconds, it evokes a dreamy and relaxing atmosphere, perfect for corporate settings.",
    "prompt3": " A soothing pop song that evokes feelings of love and relaxation, featuring a gentle blend of piano, flute, violin, and acoustic guitar. Set in the key of C major with a 4/4 time signature, the piece moves at an Andante tempo, creating a meditative and emotional atmosphere. The chord progression of G, C, F, G, and C adds to the song's calming ambiance.",
    "prompt4": "A lively and melodic rock song with a touch of pop, featuring pizzicato strings that add a playful and upbeat vibe. The piece is set in A minor and maintains a fast tempo of 148 beats per minute, with a 4/4 time signature. The chord progression of C, G, Fmaj7, C, and G repeats throughout the song, creating a catchy and energetic atmosphere that's perfect for corporate or background music.",
}

def gradio_generate(prompt, threshold, temperature, length):
    if "Amadeus-M" in model_id or "Amadeus-L" in model_id:
        encoder_choice ="large"
    else:
        encoder_choice = "base"
    text_encoder_model = 'google/flan-t5-base' if encoder_choice == 'base' else 'google/flan-t5-large'
    midi_path = generate_with_text_prompt(
        config,
        vocab,
        model,
        device,
        prompt,
        text_encoder_model,
        threshold=threshold,
        temperature=temperature,
        generation_length=length,
    )
    # === Generate corresponding WAV filename ===
    audio_path = midi_path.replace('.mid', '.wav').replace('generated', 'music/generated')
    return midi_path, audio_path

with gr.Blocks() as demo:
    gr.Markdown("# 🎵 Amadeus MIDI Generation Demo")
    gr.Markdown(
            "### 🎵 Prompt Input Guide\n"
            "Please try to include the following elements:\n"
            "- Genre (e.g. pop, electronic, ambient...)\n"
            "- Instruments (e.g. piano, guitar, drums, strings...)\n"
            "- Key (e.g. C major, F# minor...)\n"
            "- Time signature (e.g. 4/4, 3/4...)\n"
            "- Tempo (e.g. 120 BPM, Andante, Allegro...)\n"
            "- Chord progression (e.g. C, G, Am, F...)\n"
            "- Mood (e.g. happy, relaxing, motivational...)\n"
            "We recommend starting from an example prompt and then modifying it."
        )
    with gr.Row():
        prompt = gr.Textbox(label="Text Description (Prompt)", placeholder="A lively rock and electronic fusion, this song radiates happiness and energy. Distorted guitars, a rock organ, and driving drums propel the melody forward in a fast-paced 4/4 time signature. Set in the key of A major, it features a chord progression of E, D, A/G, E, and D, creating a dynamic and engaging sound that would be right at home in a video game soundtrack.")
    with gr.Row():
        threshold = gr.Slider(0.5, 1.0, 0.99, step=0.01, label="Threshold")
        temperature = gr.Slider(0.5, 3.0, 1.25, step=0.05, label="Temperature")
        length = gr.Slider(256, 3072, 1024, step=128, label="Generation Length")
    generate_btn = gr.Button("Generate MIDI 🎼")
    midi_file = gr.File(label="Download Generated MIDI File")
    audio_output = gr.Audio(label="Generated Audio Preview", type="filepath")
    generate_btn.click(fn=gradio_generate,
                       inputs=[prompt, threshold, temperature, length],
                       outputs=[midi_file, audio_output])
    gr.Markdown("### Example Prompts\n"
                "prompt1: A melodic electronic ambient song with a touch of darkness, set in the key of E major and a 4/4 time signature. Tubular bells, electric guitar, synth effects, synth pad, and oboe weave together to create an epic, space-like atmosphere. The tempo is a steady Andante, and the chord progression of A, B, and E forms the harmonic backbone of this captivating piece.\n\n"
                "prompt2: A melodic electronic song with a moderate tempo, featuring a blend of drums, piano, brass section, alto saxophone, and synth bass. The piece is set in B minor and follows a chord progression of C#m, B, A, and B. With a duration of 252 seconds, it evokes a dreamy and relaxing atmosphere, perfect for corporate settings.\n\n"
                "prompt3: A soothing pop song that evokes feelings of love and relaxation, featuring a gentle blend of piano, flute, violin, and acoustic guitar. Set in the key of C major with a 4/4 time signature, the piece moves at an Andante tempo, creating a meditative and emotional atmosphere. The chord progression of G, C, F, G, and C adds to the song's calming ambiance.\n\n"
                "prompt4: A lively and melodic rock song with a touch of pop, featuring pizzicato strings that add a playful and upbeat vibe. The piece is set in A minor and maintains a fast tempo of 148 beats per minute, with a 4/4 time signature. The chord progression of C, G, Fmaj7, C, and G repeats throughout the song, creating a catchy and energetic atmosphere that's perfect for corporate or background music."
                )   
                
    with gr.Row():
        for name, text in examples.items():
            # show text on button click
            btn = gr.Button(name)
            btn.click(lambda t=text: t, None, prompt)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)