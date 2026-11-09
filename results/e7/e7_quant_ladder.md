# E7: Quantization Ladder (vLLM)

- Device: `Tesla T4 cc=[7, 5]`
- NVFP4 supported (SM7.5/T4 expected false): `False`
- Modes requested: `['fp16', 'bnb_8bit', 'bnb_4bit']`
- TinyLlama: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Dory: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- Hard subset limit: `50`

## Results

| stage | model | mode | loops | weights_gb | tok/s | accuracy | rep_rate | max_context | status | notes/errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| tinyllama_speed_gate | TinyLlama-1.1B-Chat-v1.0 | fp16 |  |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| tinyllama_speed_gate | TinyLlama-1.1B-Chat-v1.0 | bnb_8bit |  |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| tinyllama_speed_gate | TinyLlama-1.1B-Chat-v1.0 | bnb_4bit |  |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| dory_ladder | NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | fp16 | 1 |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| dory_ladder | NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | fp16 | 4 |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| dory_ladder | NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | bnb_8bit | 1 |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| dory_ladder | NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | bnb_8bit | 4 |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| dory_ladder | NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | bnb_4bit | 1 |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |
| dory_ladder | NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | bnb_4bit | 4 |  |  |  |  |  | error | TypeError: Config() got an unexpected keyword argument 'deprecated' |

## TinyLlama Speed Gate

- `fp16`: tok/s=None weights_gb=None status=error
  - error: TypeError: Config() got an unexpected keyword argument 'deprecated'
- `bnb_8bit`: tok/s=None weights_gb=None status=error
  - error: TypeError: Config() got an unexpected keyword argument 'deprecated'
- `bnb_4bit`: tok/s=None weights_gb=None status=error
  - error: TypeError: Config() got an unexpected keyword argument 'deprecated'

## NIAH 64K Probe (Dory)

- mode=fp16 loops=1: ok=None actual_tokens=None error=
- mode=fp16 loops=4: ok=None actual_tokens=None error=
- mode=bnb_8bit loops=1: ok=None actual_tokens=None error=
- mode=bnb_8bit loops=4: ok=None actual_tokens=None error=
- mode=bnb_4bit loops=1: ok=None actual_tokens=None error=
- mode=bnb_4bit loops=4: ok=None actual_tokens=None error=
