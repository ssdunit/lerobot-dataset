import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import av
import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def to_flat_array(x, dtype=np.float32):
    """Safely flattens scalar, list, protobuf RepeatedScalarContainer, etc. into 1-D."""
    arr = np.array(x, dtype=dtype)
    return arr.reshape(-1)


def get_frame_count(local_file, topic):
    """Reads the MCAP summary index to get exact message count for a topic — no decoding needed."""
    with open(local_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        for channel_id, channel in summary.channels.items():
            if channel.topic == topic:
                return summary.statistics.channel_message_counts.get(channel_id, 0)
    return 0


def list_all_topics(local_file):
    """Diagnostic helper: prints every topic in the mcap file with its message count,
    so you can confirm state/action topic names actually match this dataset before
    running the full conversion."""
    with open(local_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        print("  --- topics found in probe file ---")
        for channel_id, channel in summary.channels.items():
            count = summary.statistics.channel_message_counts.get(channel_id, 0)
            print(f"  {channel.topic:30s} msgs={count}")
        print("  -----------------------------------")


def detect_camera_resolutions(local_file, camera_topics):
    """
    Decodes the first video frame of each camera topic to determine its actual
    (height, width) — sim cameras may differ in resolution from each other and
    from the real-robot data, so this is measured rather than assumed.
    Returns {topic: (height, width)}.
    """
    resolutions = {}
    codecs = {}

    with open(local_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic in camera_topics and channel.topic not in resolutions:
                if channel.topic not in codecs:
                    fmt = getattr(decoded, "format", "h264")
                    codec_name = "hevc" if "h265" in fmt else "h264"
                    codecs[channel.topic] = av.CodecContext.create(codec_name, 'r')

                codec = codecs[channel.topic]
                packets = codec.parse(decoded.data)
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        img = frame.to_ndarray(format='rgb24')
                        resolutions[channel.topic] = (img.shape[0], img.shape[1])  # (H, W)
                        break
                    if channel.topic in resolutions:
                        break

            if len(resolutions) == len(camera_topics):
                break

    missing = [t for t in camera_topics if t not in resolutions]
    if missing:
        raise RuntimeError(f"Could not detect resolution for camera topics: {missing}. "
                            f"Try scanning further into the file or check topic names.")
    return resolutions


def main():

    HF_USERNAME = "ssdunit"
    TASK_NAME = "set_up_chess_pieces_on_the_board"
    NEW_DATASET_ID = f"{HF_USERNAME}/abc_sim_{TASK_NAME}"
    SOURCE_REPO = "XDOF/ABC-130k"
    TEMP_DIR = "./temp_mcap_sim"
    PROGRESS_FILE = "./sim_conversion_progress.json"

    CAMERA_TOPICS = {
        "/top-camera": "observation.images.top",
        "/left-wrist-camera": "observation.images.left_wrist",
        "/right-wrist-camera": "observation.images.right_wrist",
    }

    # TODO: confirm these against the "topics found in probe file" printout below —
    # these names were carried over from the real-robot script and may not match
    # this sim dataset's actual topic names.
    STATE_ACTION_TOPICS = [
        "/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state",
        "/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action",
    ]

    os.makedirs(TEMP_DIR, exist_ok=True)

    local_cache_dir = f"\hf_cache\lerobot\{NEW_DATASET_ID}"

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

    #Episode Resume from sim_conversion_progress.json `sim_converter/sim_conversion_progress.json`
    fs = HfFileSystem()
    print(f"Scanning Hugging Face for sim '{TASK_NAME}' episodes...")
    task_path = f"datasets/{SOURCE_REPO}/data/sim/{TASK_NAME}"

    all_mcap_files = fs.glob(f"{task_path}/**/*.mcap")
    repo_files = [f.replace(f"datasets/{SOURCE_REPO}/", "") for f in all_mcap_files]
    episode_files = [f for f in repo_files if f.endswith("episode.mcap")]

    remaining_files = [f for f in episode_files if f not in completed_files]
    remaining_files=remaining_files[:50]
    print(f"Found {len(episode_files)} total episodes. "
          f"{len(completed_files)} already done, {len(remaining_files)} remaining.\n")
    if not episode_files:
        raise SystemExit(f"No episodes found under {task_path} — check TASK_NAME and repo path.")

    #Check for episode resolution before starting the conversion process
    probe_file_path = remaining_files[0] if remaining_files else episode_files[0]
    print(f"Probing resolution using: {probe_file_path}")
    probe_local_file = hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        filename=probe_file_path,
        local_dir=TEMP_DIR
    )

    resolutions_by_topic = detect_camera_resolutions(probe_local_file, list(CAMERA_TOPICS.keys()))
    for topic, (h, w) in resolutions_by_topic.items():
        print(f"  {topic} -> {CAMERA_TOPICS[topic]}: {w}x{h}")
    os.remove(probe_local_file)

    if os.path.exists(local_cache_dir):
        print(f"Found existing local dataset. Resuming {NEW_DATASET_ID}...")
        dataset = LeRobotDataset.resume(repo_id=NEW_DATASET_ID, root=local_cache_dir)
    else:
        completed_files = set()
        progress = {"completed_files": []}
        save_progress()

        features = {
            "observation.state": {"dtype": "float32", "shape": (14,)},
            "action": {"dtype": "float32", "shape": (14,)},
        }
        for topic, feature_name in CAMERA_TOPICS.items():
            h, w = resolutions_by_topic[topic]
            features[feature_name] = {"dtype": "video", "shape": (3, h, w)}

        print(f"Initializing empty LeRobot dataset: {NEW_DATASET_ID}")
        dataset = LeRobotDataset.create(
            repo_id=NEW_DATASET_ID,
            fps=30,
            features=features,
            root=local_cache_dir,
        )

    #IMPORTANT TQDM CODE
    #TODO:Help Better tqdm with constant episode updates as well as add tqdm for pause phase during svt conversion process
    episode_bar = tqdm(remaining_files, desc="Episodes", unit="ep")

    for idx, file_path in enumerate(episode_bar):
        episode_bar.set_postfix_str(file_path.split("/")[-2][:20])

        local_file = hf_hub_download(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            filename=file_path,
            local_dir=TEMP_DIR
        )

        est_total = get_frame_count(local_file, "/top-camera")

        # State buffers (position only, per arm + gripper — matches your real-robot script's structure)
        l_joints_state = np.zeros(6, dtype=np.float32)
        l_grip_state = np.zeros(1, dtype=np.float32)
        r_joints_state = np.zeros(6, dtype=np.float32)
        r_grip_state = np.zeros(1, dtype=np.float32)

        # Action buffers , real actions this time, not a copy of state
        l_joints_action = np.zeros(6, dtype=np.float32)
        l_grip_action = np.zeros(1, dtype=np.float32)
        r_joints_action = np.zeros(6, dtype=np.float32)
        r_grip_action = np.zeros(1, dtype=np.float32)

        task_label = TASK_NAME  
        codecs = {}  # one codec instance per camera topic ,can't share across streams
        image_buffers = {topic: None for topic in CAMERA_TOPICS}

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
                        l_joints_state = to_flat_array(decoded.position)
                    elif channel.topic == "/left-ee-state":
                        l_grip_state = to_flat_array(decoded.position)
                    elif channel.topic == "/right-arm-state":
                        r_joints_state = to_flat_array(decoded.position)
                    elif channel.topic == "/right-ee-state":
                        r_grip_state = to_flat_array(decoded.position)

                    elif channel.topic == "/left-arm-action":
                        l_joints_action = to_flat_array(decoded.position)
                    elif channel.topic == "/left-ee-action":
                        l_grip_action = to_flat_array(decoded.position)
                    elif channel.topic == "/right-arm-action":
                        r_joints_action = to_flat_array(decoded.position)
                    elif channel.topic == "/right-ee-action":
                        r_grip_action = to_flat_array(decoded.position)

                    elif channel.topic in CAMERA_TOPICS:
                        if channel.topic not in codecs:
                            fmt = getattr(decoded, "format", "h264")
                            codec_name = "hevc" if "h265" in fmt else "h264"
                            codecs[channel.topic] = av.CodecContext.create(codec_name, 'r')

                        codec = codecs[channel.topic]
                        packets = codec.parse(decoded.data)

                        for packet in packets:
                            frames = codec.decode(packet)
                            for frame in frames:
                                img = frame.to_ndarray(format='rgb24')
                                image_buffers[channel.topic] = np.transpose(img, (2, 0, 1))
                                break

                        """Frame boundary keyed off /top-camera producing a new frame — same pattern as the real-robot script's single-camera design. Other
                        cameras' most recent decoded frame is reused if they haven't updated at exactly the same message cadence."""
                        if channel.topic == "/top-camera" and image_buffers["/top-camera"] is not None:
                            if any(image_buffers[t] is None for t in CAMERA_TOPICS):
                                # Not all cameras have produced a first frame yet ,skip until they have
                                frame_bar.update(1)
                                continue

                            state = np.concatenate([
                                l_joints_state, l_grip_state, r_joints_state, r_grip_state
                            ])
                            action = np.concatenate([
                                l_joints_action, l_grip_action, r_joints_action, r_grip_action
                            ])
                            assert state.shape[0] == 14, f"Bad state shape {state.shape} at {file_path}"
                            assert action.shape[0] == 14, f"Bad action shape {action.shape} at {file_path}"

                            frame_data = {
                                "observation.state": state,
                                "action": action,
                                "task": task_label,
                            }
                            for topic, feature_name in CAMERA_TOPICS.items():
                                img = image_buffers[topic]
                                expected_h, expected_w = resolutions_by_topic[topic]
                                if (
                                    img is None
                                    or img.shape != (3, expected_h, expected_w)
                                    or img.dtype != np.uint8
                                ):
                                    raise RuntimeError(
                                        f"Corrupt/invalid frame for {topic} at frame "
                                        f"{frame_bar.n} in episode {file_path}: "
                                        f"shape={None if img is None else img.shape}, "
                                        f"dtype={None if img is None else img.dtype}, "
                                        f"expected shape=(3, {expected_h}, {expected_w})"
                                    )
                                frame_data[feature_name] = img

                            dataset.add_frame(frame_data)
                            frame_bar.update(1)

            frame_bar.close()
            dataset.save_episode()

            completed_files.add(file_path)
            save_progress()

        except Exception as e:
            frame_bar.close()
            print(f"\Failed on episode {file_path}: {e}")
            print("Progress file NOT updated for this episode ,it will be retried on next run.")
            if os.path.exists(local_file):
                os.remove(local_file)
            raise

        os.remove(local_file)

    episode_bar.close()

    #Final Push to hugging face
    
    print("Consolidating dataset (computing statistics)...")
    dataset.finalize()

    print(f"Final push to {NEW_DATASET_ID}...")
    dataset.push_to_hub()

    print("Your sim dataset is ready for lerobot-train.")


if __name__ == "__main__":
    main()