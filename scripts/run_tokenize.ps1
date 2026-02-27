param(
  [string]$InputRoot = "data/raw_audio",
  [string]$OutputRoot = "data/tokens",
  [string]$Device = "auto",
  [int]$MaxFiles = 0
)

$maxFilesArgs = @()
if ($MaxFiles -gt 0) {
  $maxFilesArgs = @("--max-files", $MaxFiles)
}

python -m src.tokenizer.moss_tokenize `
  --input-root $InputRoot `
  --output-root $OutputRoot `
  --device $Device `
  @maxFilesArgs

