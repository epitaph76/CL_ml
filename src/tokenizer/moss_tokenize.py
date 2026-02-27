from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TokenizeConfig:
    input_root: Path
    output_root: Path
    model_id: str
    device: str
    chunk_seconds: float | None
    extensions: tuple[str, ...]
    max_files: int | None
    overwrite: bool
    save_format: str


def _parse_args() -> TokenizeConfig:
    parser = argparse.ArgumentParser(
        description="Offline tokenization: audio files -> MOSS discrete codes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root folder with audio files (recursive scan).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/tokens"),
        help="Where to save token files.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        help="HuggingFace model id for MOSS tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run tokenization on.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Optional chunk length (seconds). If omitted, whole track is tokenized once.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp3", ".wav", ".flac", ".ogg", ".m4a"],
        help="Audio file extensions to include.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate output files if they already exist.",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["pt", "npz"],
        default="pt",
        help="Token storage format.",
    )
    args = parser.parse_args()

    return TokenizeConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        model_id=args.model_id,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        extensions=tuple(sorted(ext.lower() for ext in args.extensions)),
        max_files=args.max_files,
        overwrite=args.overwrite,
        save_format=args.save_format,
    )


def _sanitize_part(part: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", part)


def build_track_id(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root).with_suffix("")
    pieces = [_sanitize_part(piece) for piece in rel.parts]
    return "__".join(pieces)


def discover_audio_files(
    input_root: Path, extensions: Sequence[str], max_files: int | None
) -> list[Path]:
    ext_set = set(extensions)
    audio_files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in ext_set
    ]
    audio_files.sort()
    if max_files is not None:
        return audio_files[:max_files]
    return audio_files


def _resolve_device(device_flag: str):
    import torch

    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_flag)


def _load_moss(model_id: str, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def _load_audio(path: Path, target_sr: int):
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform.squeeze(0).to(torch.float32), target_sr


def _split_waveform(waveform, sample_rate: int, chunk_seconds: float | None):
    if not chunk_seconds:
        return [waveform]
    chunk_samples = int(sample_rate * chunk_seconds)
    if chunk_samples <= 0:
        return [waveform]
    chunks = []
    for start in range(0, waveform.shape[0], chunk_samples):
        end = min(start + chunk_samples, waveform.shape[0])
        if end - start <= 0:
            continue
        chunks.append(waveform[start:end])
    return chunks or [waveform]


def _save_payload(out_path: Path, payload: dict, save_format: str) -> None:
    import numpy as np
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "pt":
        torch.save(payload, out_path.with_suffix(".pt"))
        return

    tokens = payload["tokens"]
    token_array = tokens.detach().cpu().numpy() if hasattr(tokens, "detach") else tokens
    metadata = {k: v for k, v in payload.items() if k != "tokens"}
    np.savez_compressed(
        out_path.with_suffix(".npz"),
        tokens=token_array,
        metadata=json.dumps(metadata, ensure_ascii=True),
    )


def _iter_missing_outputs(
    audio_files: Iterable[Path],
    config: TokenizeConfig,
) -> list[Path]:
    selected: list[Path] = []
    for path in audio_files:
        track_id = build_track_id(path, config.input_root)
        out_stem = config.output_root / track_id
        target_path = out_stem.with_suffix(f".{config.save_format}")
        if not config.overwrite and target_path.exists():
            continue
        selected.append(path)
    return selected


def _extract_codes(encode_output):
    """Normalize MOSS output into a tensor with code shape [num_quantizers, seq_len]."""
    import torch

    if isinstance(encode_output, tuple):
        code_tensor = encode_output[0]
    elif hasattr(encode_output, "audio_codes"):
        code_tensor = encode_output.audio_codes
    else:
        raise RuntimeError("Unsupported `model.encode` output type.")

    if not isinstance(code_tensor, torch.Tensor):
        raise RuntimeError("`model.encode` returned non-tensor audio codes.")
    if code_tensor.dim() != 3:
        raise RuntimeError(f"Expected 3D codes tensor, got shape: {tuple(code_tensor.shape)}")

    # MOSS internal layout is usually [num_quantizers, batch_size, seq_len].
    # We process one track at a time, so one of the first two dims must be batch=1.
    if code_tensor.shape[1] == 1:
        tokens = code_tensor[:, 0, :]
    elif code_tensor.shape[0] == 1:
        tokens = code_tensor[0, :, :]
    else:
        raise RuntimeError(
            "Cannot infer batch dimension from code tensor shape. "
            f"Got shape={tuple(code_tensor.shape)}"
        )
    return tokens


def run_tokenization(config: TokenizeConfig) -> None:
    import torch
    from tqdm import tqdm

    if not config.input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {config.input_root}")

    device = _resolve_device(config.device)
    audio_files = discover_audio_files(config.input_root, config.extensions, config.max_files)
    if not audio_files:
        raise RuntimeError(
            f"No files found in {config.input_root} for extensions: {config.extensions}"
        )

    audio_files = _iter_missing_outputs(audio_files, config)
    if not audio_files:
        print("Nothing to tokenize: all files already processed.")
        return

    print(f"Found {len(audio_files)} file(s) to tokenize.")
    print(f"Device: {device}")
    processor, model = _load_moss(config.model_id, device)
    target_sr = int(processor.feature_extractor.sampling_rate)
    print(f"MOSS target sample rate: {target_sr}")

    total_chunks = 0
    for audio_path in tqdm(audio_files, desc="Tokenizing", unit="track"):
        track_id = build_track_id(audio_path, config.input_root)
        waveform, _ = _load_audio(audio_path, target_sr)
        chunks = _split_waveform(waveform, target_sr, config.chunk_seconds)
        duration_sec = float(waveform.shape[0] / target_sr)

        for chunk_index, chunk_waveform in enumerate(chunks):
            inputs = processor(
                [chunk_waveform.numpy()],
                sampling_rate=target_sr,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(device)
            padding_mask = (
                inputs.padding_mask.to(device)
                if hasattr(inputs, "padding_mask")
                else torch.ones_like(input_values, dtype=torch.long, device=device)
            )
            with torch.inference_mode():
                codes = model.encode(
                    input_values=input_values,
                    padding_mask=padding_mask,
                    return_dict=False,
                )

            tokens = _extract_codes(codes).detach().cpu()
            output_stem = config.output_root / track_id
            if config.chunk_seconds:
                output_stem = config.output_root / f"{track_id}__chunk{chunk_index:04d}"

            payload = {
                "track_id": track_id,
                "source_path": str(audio_path),
                "sample_rate": target_sr,
                "duration_sec": duration_sec,
                "chunk_index": chunk_index,
                "num_chunks": len(chunks),
                "token_shape": list(tokens.shape),
                "tokens": tokens,
            }
            _save_payload(output_stem, payload, config.save_format)
            total_chunks += 1

    print(f"Done. Saved {total_chunks} token file(s) into: {config.output_root}")


def main() -> None:
    config = _parse_args()
    run_tokenization(config)


if __name__ == "__main__":
    main()
