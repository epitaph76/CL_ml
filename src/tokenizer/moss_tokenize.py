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
    fail_fast: bool


def _parse_args() -> TokenizeConfig:
    parser = argparse.ArgumentParser(
        description="Offline tokenization: audio files -> MOSS discrete codes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Folder with audio files (recursive) or a single audio file path.",
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
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed file instead of skipping and continuing.",
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
        fail_fast=args.fail_fast,
    )


def _sanitize_part(part: str) -> str:
    cleaned = re.sub(r"[^\w-]+", "_", part, flags=re.UNICODE).strip("_")
    return cleaned if cleaned else "untitled"


def build_track_id(path: Path, input_root: Path) -> str:
    try:
        rel = path.relative_to(input_root).with_suffix("")
    except ValueError:
        rel = path.with_suffix("")
    pieces = [_sanitize_part(piece) for piece in rel.parts]
    return "__".join(pieces)


def discover_audio_files(
    input_root: Path, extensions: Sequence[str], max_files: int | None
) -> tuple[list[Path], Path]:
    ext_set = set(extensions)
    if input_root.is_file():
        if input_root.suffix.lower() not in ext_set:
            raise RuntimeError(
                f"File extension {input_root.suffix} is not in allowed extensions: {tuple(sorted(ext_set))}"
            )
        audio_files = [input_root]
        scan_root = input_root.parent
    else:
        audio_files = [
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in ext_set
        ]
        scan_root = input_root

    audio_files.sort()
    if max_files is not None:
        audio_files = audio_files[:max_files]
    return audio_files, scan_root


def _resolve_device(device_flag: str):
    import torch

    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_flag)


def _load_moss(model_id: str, device):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    return model


def _load_audio(path: Path, target_sr: int):
    import torch
    import torchaudio

    try:
        waveform, sample_rate = torchaudio.load(path)
    except Exception as torchaudio_exc:
        # Fallback path for environments where torchaudio backend lacks mp3 support.
        try:
            import librosa
        except Exception as import_exc:
            raise RuntimeError(
                f"Failed to load audio via torchaudio for {path}. "
                f"Also failed to import librosa fallback: {import_exc}"
            ) from torchaudio_exc

        try:
            audio, _ = librosa.load(path, sr=target_sr, mono=True)
        except Exception as librosa_exc:
            raise RuntimeError(
                f"Failed to load audio file: {path}\n"
                f"torchaudio error: {torchaudio_exc}\n"
                f"librosa error: {librosa_exc}"
            ) from librosa_exc

        waveform = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
        sample_rate = target_sr

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
    scan_root: Path,
) -> list[Path]:
    selected: list[Path] = []
    for path in audio_files:
        track_id = build_track_id(path, scan_root)
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
    audio_files, scan_root = discover_audio_files(
        config.input_root, config.extensions, config.max_files
    )
    if not audio_files:
        raise RuntimeError(
            f"No files found in {config.input_root} for extensions: {config.extensions}"
        )

    audio_files = _iter_missing_outputs(audio_files, config, scan_root)
    if not audio_files:
        print("Nothing to tokenize: all files already processed.")
        return

    print(f"Found {len(audio_files)} file(s) to tokenize.")
    print(f"Device: {device}")
    model = _load_moss(config.model_id, device)
    target_sr = int(
        getattr(model, "sampling_rate", None)
        or getattr(getattr(model, "config", None), "sampling_rate", None)
        or 24000
    )
    print(f"MOSS target sample rate: {target_sr}")

    total_chunks = 0
    failed_files: list[tuple[Path, str]] = []
    for audio_path in tqdm(audio_files, desc="Tokenizing", unit="track"):
        try:
            track_id = build_track_id(audio_path, scan_root)
            waveform, _ = _load_audio(audio_path, target_sr)
            chunks = _split_waveform(waveform, target_sr, config.chunk_seconds)
            duration_sec = float(waveform.shape[0] / target_sr)

            for chunk_index, chunk_waveform in enumerate(chunks):
                input_values = chunk_waveform.unsqueeze(0).unsqueeze(0).to(device)
                padding_mask = torch.ones(
                    (1, chunk_waveform.shape[0]), dtype=torch.long, device=device
                )
                with torch.inference_mode():
                    codes = model.encode(
                        input_values=input_values,
                        padding_mask=padding_mask,
                        return_dict=True,
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
        except Exception as exc:
            failed_files.append((audio_path, str(exc)))
            print(f"[WARN] Failed to tokenize {audio_path}: {exc}")
            if config.fail_fast:
                raise

    print(f"Done. Saved {total_chunks} token file(s) into: {config.output_root}")
    if failed_files:
        print(f"Skipped {len(failed_files)} file(s) due to errors.")
        for path, reason in failed_files[:10]:
            print(f"  - {path}: {reason}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    if total_chunks == 0 and failed_files:
        raise RuntimeError("All files failed during tokenization. See warnings above.")


def main() -> None:
    config = _parse_args()
    run_tokenization(config)


if __name__ == "__main__":
    main()
