# AIaudioEQmaster

> Educational project for audio mastering pipeline study and customization.
> 本项目仅供学习与技术交流使用，请勿用于侵权、非法或误导性用途。

## Overview | 项目简介

**AIaudioEQmaster** is a modular audio mastering project with:

- A lightweight ML suggestion layer (genre-based target hints)
- A CPU mastering pipeline (`src/cpu`)
- An accelerated experimental CPU pipeline (`src/cpu_test`)
- Configurable mastering parameters: input path, genre profile, style, loudness

**AIaudioEQmaster** 是一个可客制化的母带处理基础项目，包含：

- 轻量模型建议层（按风格给参数建议）
- 标准 CPU 处理链（`src/cpu`）
- 加速实验链路（`src/cpu_test`）
- 可配置参数：输入文件、风格档案、EQ 风格、响度模式

## Disclaimer | 免责声明

- For learning/research only.
- No warranty for commercial mastering quality.
- Please ensure legal rights for all input audio.

- 仅用于学习、研究与交流。
- 不保证满足商业母带制作标准。
- 请确保你对输入音频拥有合法使用权。

## Project Structure | 目录结构

```text
AIaudioEQmaster/
├─ main.py
├─ requirements.txt
├─ pyproject.toml
├─ resources/
│  ├─ model/
│  ├─ profiles/
│  └─ secured_genres/
├─ scripts/
│  └─ run_mastering.ps1
└─ src/
   ├─ cli.py
   ├─ mastering.py
   ├─ test_model.py
   ├─ lib.py
   ├─ common/
   │  └─ paths.py
   ├─ cpu/
   │  └─ (baseline mastering modules)
   └─ cpu_test/
      └─ (accelerated experiment modules)
```

## Installation | 安装

### Option A: uv

```bash
uv venv
# Windows PowerShell
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Option B: pip

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage | 使用方法

### 1) Basic CLI | 基础命令

```bash
python main.py "<input_audio>" "<output_audio.wav>" --genre Pop --eq-profile Fusion --loudness loud
```

Parameters:

- `--genre`: style profile, default recommended `Pop`
- `--eq-profile`: `Neutral | Warm | Bright | Fusion` (default recommended `Fusion`)
- `--loudness`: `soft | dynamic | normal | loud` (default recommended `loud`)

参数说明：

- `--genre`：风格档案，推荐默认 `Pop`
- `--eq-profile`：`Neutral | Warm | Bright | Fusion`（推荐默认 `Fusion`）
- `--loudness`：`soft | dynamic | normal | loud`（推荐默认 `loud`）

### 2) PowerShell helper | PowerShell 脚本

```powershell
./scripts/run_mastering.ps1 -Input "D:\music\input.flac" -Output "D:\music\output.wav" -Genre Pop -Style Fusion -Loudness loud
```

## Notes | 说明

- `src/cpu` emphasizes stability/consistency.
- `src/cpu_test` emphasizes speed and iterative optimization.
- You can A/B compare outputs in DAW tools (LUFS, True Peak, spectrum, transient behavior).

- `src/cpu` 更偏稳定一致。
- `src/cpu_test` 更偏速度与实验迭代。
- 建议在 DAW 里做 A/B 对比（LUFS、True Peak、频谱、瞬态）。

## TODO

- [ ] Add GPU acceleration support (CUDA)
- [ ] Add objective quality evaluation report (LUFS/TP/DR/spectral diff)
- [ ] Add batch processing CLI mode
- [ ] Add preset management and custom profile editor
- [ ] Add automated regression tests for audio metrics
- [ ] Add optional real-time/low-latency preview path

## License

See [LICENSE](./LICENSE).
