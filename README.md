# abc2Lerobot Dataset
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

### Simulation 

1. To determine the channel topics `sim_converter/check.py`

```shell
cd sim_converter
python3 check.py
```

2. To determine the camera topics and resolution (`sim_converter/camera_resolution.py`)

```shell
cd sim_converter
python3 camera_resolution.py
```

3. To use the converter (`sim_converter/sim_converter.py`)

```shell
cd sim_converter
python3 sim_converter.py
```

### Real Robot Dataset (abc2lerobot)

1. To determine the channel topics (Important:Ensure the converter script includes the topics that exist in the .mcap file) [`robot_data_converter/check.py`]

```shell
cd robot_data_converter
python3 check.py
```

2. To use the converter [`robot_data_converter/converter.py`]

```shell
cd robot_data_converter
python3 converter.py
```
