import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import av
import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub import login
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

"""Use check.py , to check existing topic list in channel"""

#Converts eg. (14,14) to (14,)
def to_flat_array(x, dtype=np.float32):
    """Safely flattens scalar, list, protobuf RepeatedScalarContainer, etc. into 1-D."""
    arr = np.array(x, dtype=dtype)
    return arr.reshape(-1)

#Total frame count for tqdm
def get_frame_count(local_file, topic="/top-left-camera"):
    """Reads the MCAP summary index to get exact message count for a topic, no decoding needed."""
    with open(local_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        for channel_id, channel in summary.channels.items():
            if channel.topic == topic:
                return summary.statistics.channel_message_counts.get(channel_id, 0)
    return 0

HF_USERNAME = "ssdunit"
TASK_NAME = "dress_the_teddy_bear"
NEW_DATASET_ID = f"{HF_USERNAME}/abc_dress_the_teddy_bear"
SOURCE_REPO = "XDOF/ABC-130k"
TEMP_DIR = "./temp_mcap"
PROGRESS_FILE = "./conversion_progress.json"
PUSH_EVERY_N_EPISODES = 1   # push to hub after every N completed episodes

os.makedirs(TEMP_DIR, exist_ok=True)

local_cache_dir = f"\hf_cache\lerobot\{NEW_DATASET_ID}"

#Progress log(Conversion_progress.json)
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
    completed_files = set(progress.get("completed_files", []))
    print(f"Resuming: found progress file with {len(completed_files)} episodes already done.")
else:
    completed_files = set()
    progress = {"completed_files": []}
    print("No progress file found. Starting fresh.")

def save_progress():
    progress["completed_files"] = list(completed_files)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

#Initialize new dataset on hf or resume preexisting
if completed_files and os.path.exists(local_cache_dir):
    print(f"Found existing local dataset with prior progress. Re-opening {NEW_DATASET_ID}")
    dataset = LeRobotDataset.resume(repo_id=NEW_DATASET_ID,root=local_cache_dir)
else:
    import shutil
    if os.path.exists(local_cache_dir):
        print("Found old dataset cache but no progress log, deleting to start fresh...")
        shutil.rmtree(local_cache_dir)
    completed_files = set()
    progress = {"completed_files": []}

    print(f"Initializing empty LeRobot dataset: {NEW_DATASET_ID}")
    dataset = LeRobotDataset.create(
        repo_id=NEW_DATASET_ID,
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (14,)},
            "action": {"dtype": "float32", "shape": (14,)},
            "observation.images.top": {"dtype": "video", "shape": (3, 1200, 1920)},
        }
    )

fs = HfFileSystem()
print(f"Scanning Hugging Face for '{TASK_NAME}' episodes...")
task_path = f"datasets/{SOURCE_REPO}/data/train/{TASK_NAME}"

all_mcap_files = fs.glob(f"{task_path}/**/*.mcap")
repo_files = [f.replace(f"datasets/{SOURCE_REPO}/", "") for f in all_mcap_files]
episode_files = [f for f in repo_files if f.endswith("episode.mcap")]

remaining_files = [f for f in episode_files if f not in completed_files]
print(f"Found {len(episode_files)} total episodes. "
      f"{len(completed_files)} already done, {len(remaining_files)} remaining.\n")


episode_bar = tqdm(remaining_files, desc="Episodes", unit="ep")
episodes_since_push = 0

for idx, file_path in enumerate(episode_bar):
    episode_bar.set_postfix_str(file_path.split("/")[-2][:20])

    local_file = hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        filename=file_path,
        local_dir=TEMP_DIR
    )

    est_total = get_frame_count(local_file)
    #14 DOF , Initialized zero arrays for each joint and gripper
    l_joints = np.zeros(6, dtype=np.float32)
    l_grip = np.zeros(1, dtype=np.float32)
    r_joints = np.zeros(6, dtype=np.float32)
    r_grip = np.zeros(1, dtype=np.float32)

    codec = None

    frame_bar = tqdm(
        desc="  Ep frames",
        unit="frame",
        leave=False,
        total=est_total if est_total > 0 else None,
    )

    try:
        with open(local_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            for schema, channel, message, decoded in reader.iter_decoded_messages():

                if channel.topic == "/left-arm-state":
                    l_joints = to_flat_array(decoded.position)
                elif channel.topic == "/left-ee-state":
                    l_grip = to_flat_array(decoded.position)
                elif channel.topic == "/right-arm-state":
                    r_joints = to_flat_array(decoded.position)
                elif channel.topic == "/right-ee-state":
                    r_grip = to_flat_array(decoded.position)

                elif channel.topic == "/top-left-camera":
                    if codec is None:
                        fmt = getattr(decoded, "format", "h264")
                        codec_name = "hevc" if "h265" in fmt else "h264"
                        codec = av.CodecContext.create(codec_name, 'r')

                    packets = codec.parse(decoded.data)
                    image_array = None

                    for packet in packets:
                        frames = codec.decode(packet)
                        for frame in frames:
                            img = frame.to_ndarray(format='rgb24')
                            image_array = np.transpose(img, (2, 0, 1))
                            break

                    if image_array is None:
                        frame_bar.update(1)
                        continue

                    state = np.concatenate([l_joints, l_grip, r_joints, r_grip])
                    assert state.shape[0] == 14, f"Bad state shape {state.shape} at {file_path}"

                    dataset.add_frame({
                        "observation.state": state,
                        "action": state,
                        "observation.images.top": image_array,
                        "task": TASK_NAME
                    })

                    frame_bar.update(1)

        frame_bar.close()
        dataset.save_episode()

        dataset.finalize()

        completed_files.add(file_path) #add to conversion progress to prevent save loss
        save_progress()

    except Exception as e:
        frame_bar.close()
        print(f"\nFailed on episode {file_path}: {e}")
        print("Progress file NOT updated for this episode — it will be retried on next run.")
        if os.path.exists(local_file):
            os.remove(local_file)
        raise

    os.remove(local_file)
    episodes_since_push += 1

    # Periodic push so a session timeout doesn't lose everything
    if episodes_since_push >= PUSH_EVERY_N_EPISODES:
        print(f"\nPushing checkpoint to hub after {len(completed_files)} total episodes...")
        dataset.push_to_hub()

        episodes_since_push = 0

episode_bar.close()

#Final Push
print("Consolidating dataset (computing statistics)...")
dataset.finalize()

print(f"Final push to {NEW_DATASET_ID}")
dataset.push_to_hub()

print("Dataset saved")