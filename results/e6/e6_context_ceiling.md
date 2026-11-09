# E6 Context Ceiling (NIAH)

- Generated: `2026-11-09T10:34:31.578077+00:00`
- Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- Contexts: `[32768, 65536, 131072]`
- Modes: `['last_loop', 'per_loop']`
- Loops: `[1, 4]`
- Gen tokens: `64`

## Results

| context | mode | loops | status | accuracy | correct/total | elapsed_s |
|---:|---|---:|---|---:|---:|---:|
| 32768 | last_loop | 1 | error |  | 0/0 | 46.31 |

<details><summary>Error: context=32768 mode=last_loop loops=1</summary>

```text
The decoder prompt (length 32768) plus the number of requested output tokens (at least 1) is longer than the maximum model length of 32768. Make sure that `max_model_len` is no smaller than the number of text tokens (prompt + requested output tokens).
```
</details>

| 32768 | last_loop | 4 | error |  | 0/0 | 44.64 |

<details><summary>Error: context=32768 mode=last_loop loops=4</summary>

```text
The decoder prompt (length 32768) plus the number of requested output tokens (at least 1) is longer than the maximum model length of 32768. Make sure that `max_model_len` is no smaller than the number of text tokens (prompt + requested output tokens).
```
</details>

| 32768 | per_loop | 1 | error |  | 0/0 | 45.08 |

<details><summary>Error: context=32768 mode=per_loop loops=1</summary>

```text
The decoder prompt (length 32768) plus the number of requested output tokens (at least 1) is longer than the maximum model length of 32768. Make sure that `max_model_len` is no smaller than the number of text tokens (prompt + requested output tokens).
```
</details>

| 32768 | per_loop | 4 | error |  | 0/0 | 45.15 |

<details><summary>Error: context=32768 mode=per_loop loops=4</summary>

```text
The decoder prompt (length 32768) plus the number of requested output tokens (at least 1) is longer than the maximum model length of 32768. Make sure that `max_model_len` is no smaller than the number of text tokens (prompt + requested output tokens).
```
</details>

| 65536 | last_loop | 1 | error |  | 0/0 | 40.45 |

<details><summary>Error: context=65536 mode=last_loop loops=1</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 65536 | last_loop | 4 | error |  | 0/0 | 40.70 |

<details><summary>Error: context=65536 mode=last_loop loops=4</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 65536 | per_loop | 1 | error |  | 0/0 | 40.78 |

<details><summary>Error: context=65536 mode=per_loop loops=1</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 65536 | per_loop | 4 | error |  | 0/0 | 40.86 |

<details><summary>Error: context=65536 mode=per_loop loops=4</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 131072 | last_loop | 1 | error |  | 0/0 | 40.89 |

<details><summary>Error: context=131072 mode=last_loop loops=1</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 131072 | last_loop | 4 | error |  | 0/0 | 40.99 |

<details><summary>Error: context=131072 mode=last_loop loops=4</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 131072 | per_loop | 1 | error |  | 0/0 | 41.02 |

<details><summary>Error: context=131072 mode=per_loop loops=1</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 131072 | per_loop | 4 | error |  | 0/0 | 41.30 |

<details><summary>Error: context=131072 mode=per_loop loops=4</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>


## Memory Diagnostic (64K)

- Mode: `last_loop`

| loops | status | allocated | reserved | max_allocated | max_reserved |
|---:|---|---:|---:|---:|---:|
| 1 | error | 0 | 0 | 0 | 0 |

<details><summary>Diagnostic error (loops=1)</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

| 4 | error | 0 | 0 | 0 | 0 |

<details><summary>Diagnostic error (loops=4)</summary>

```text
Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
</details>

