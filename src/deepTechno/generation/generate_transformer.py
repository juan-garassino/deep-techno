"""Music Transformer generation: beam search or random sampling."""
import os
import random

import torch

from deepTechno.data.dataset import MaestroDataset
from deepTechno.data.tokenizer import encode_midi, decode_midi
from deepTechno.model.constants import TORCH_LABEL_TYPE
from deepTechno.model.transformer import MusicTransformer
from deepTechno.utils.device import get_device, use_cuda


def load_model(weights_path: str, n_layers=6, num_heads=8, d_model=512,
               dim_feedforward=1024, max_sequence=2048, rpr=True) -> MusicTransformer:
    model = MusicTransformer(
        n_layers=n_layers, num_heads=num_heads, d_model=d_model,
        dim_feedforward=dim_feedforward, max_sequence=max_sequence, rpr=rpr,
    ).to(get_device())
    model.load_state_dict(torch.load(weights_path, map_location=get_device()))
    model.eval()
    return model


def generate_from_primer(model: MusicTransformer, primer_tokens: list,
                          target_seq_length: int = 1024, beam: int = 0,
                          out_file: str = None):
    """
    Generate from a list of primer event tokens.

    Args:
        model:             trained MusicTransformer
        primer_tokens:     list of int event tokens used as primer
        target_seq_length: max tokens to generate
        beam:              0 = random sampling, >0 = beam width
        out_file:          if given, write output MIDI to this path

    Returns:
        generated token list
    """
    primer = torch.tensor(primer_tokens, dtype=TORCH_LABEL_TYPE, device=get_device())
    with torch.no_grad():
        gen = model.generate(primer=primer, target_seq_length=target_seq_length, beam=beam)
    tokens = gen[0].cpu().numpy()

    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        decode_midi(tokens, file_path=out_file)
        print(f"Saved → {out_file}")

    return tokens.tolist()


def generate_from_file(model: MusicTransformer, primer_file: str,
                       num_primer: int = 256, target_seq_length: int = 1024,
                       beam: int = 0, out_file: str = None):
    """Generate using a MIDI file as primer."""
    raw = encode_midi(primer_file)
    primer_tokens = raw[:num_primer]
    return generate_from_primer(model, primer_tokens, target_seq_length, beam, out_file)


def generate_from_dataset(model: MusicTransformer, csv_path: str,
                          idx: int = None, num_primer: int = 256,
                          target_seq_length: int = 1024, beam: int = 0,
                          out_dir: str = "./results/generated"):
    """Pick a random (or indexed) file from a dataset CSV and generate from it."""
    dataset = MaestroDataset(csv_path, max_seq=num_primer, random_seq=False)
    idx = idx if idx is not None else random.randrange(len(dataset))
    x, _ = dataset[idx]
    primer_tokens = x.tolist()

    os.makedirs(out_dir, exist_ok=True)
    mode = "beam" if beam > 0 else "rand"
    out_file = os.path.join(out_dir, f"{mode}.mid")

    return generate_from_primer(model, primer_tokens, target_seq_length, beam, out_file)
