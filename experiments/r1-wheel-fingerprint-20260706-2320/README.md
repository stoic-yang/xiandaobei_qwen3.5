# r1-wheel-fingerprint-20260706-2320

- Intent: verify what code was actually measured by `pip install vllm_cscc_competition/dist/*.whl`.
- Method: compare SHA256 of selected files across remote source git commits, the wheel zip contents, and installed `/usr/local/lib/python3.10/dist-packages/vllm`.
- Raw evidence: `raw/fingerprint.json`.
- Remote repo: `/public/home/xdzs2026_c166/vllm_cscc_competition`
- Repo HEAD: `d29e9db3ffa01b701346445c6e62fe963f6c17b1`
- Wheel: `/public/home/xdzs2026_c166/vllm_cscc_competition/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`
- Wheel SHA256: `a0f09295a60dc1e5f4f7e9a096f540f29165168047c3caaf37233b6e4cb8cfde`

## Verdict

The measured competition wheel is **not equivalent to source HEAD `d29e9db3`** for the d29 attention change.

`vllm/v1/attention/ops/triton_unified_attention.py`:

| source | SHA256 | matches |
|---|---|---|
| wheel | `8e8d393c4d551547de397859462ad7c3750230841458e658d73381dbf3f59005` | `33323a1` through `a55f3c3` |
| site-packages | `8e8d393c4d551547de397859462ad7c3750230841458e658d73381dbf3f59005` | `33323a1` through `a55f3c3` |
| source `d29e9db3` | `acf4b51ba9250014a08ae54f91c775d3764cbf350856a077e484a76b52cba3f8` | source only |

Implication: `guard-d29e9db3-20260706-2005` measured the installed competition wheel, whose attention path is `a55f3c3`-equivalent for the d29 file. It must not be cited as a real measurement of the d29 GQA attention path unless a rebuilt wheel or explicit runtime overlay is used.

## Selected File Matches

| file | wheel/site matches |
|---|---|
| `qwen3_5.py` | `fde463d`, `a55f3c3`, `d29e9db3` |
| `qwen3_next.py` | `a55f3c3`, `d29e9db3` |
| `activation.py` | `293566c`, `fde463d`, `a55f3c3`, `d29e9db3` |
| `chunk.py` | `293566c`, `fde463d`, `a55f3c3`, `d29e9db3` |
| `chunk_o.py` | `293566c`, `fde463d`, `a55f3c3`, `d29e9db3` |
| `triton_unified_attention.py` | `33323a1`, `993a944`, `0ba4953`, `293566c`, `fde463d`, `a55f3c3` |

`version.py` in the wheel is generated metadata and does not match the source commit file hash.

