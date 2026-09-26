# Lerobot Dataset
## Installations

1. Install Cmake,FFmpeg,build-essential (Can skip if already present)

```shell
sudo apt-get update && sudo apt-get install -y ffmpeg cmake build-essential
```

2. Create a virtual environment and install the requirements

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts use

1. To determine the channel topics `check.py`

```shell
python3 check.py
```

2. To use the converter (`converter.py`)

```shell
python3 converter.py
```
