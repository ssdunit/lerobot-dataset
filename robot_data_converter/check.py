from huggingface_hub import hf_hub_download
from mcap.reader import make_reader

SOURCE_REPO = "XDOF/ABC-130k"
TASK_NAME = "dress_the_teddy_bear"
TEMP_DIR = "./temp_mcap"
Episode_Name = "" #Put the episode folder string name from XDOF/ABC-130k/YourTask/train/

#Test on preexisting dataset episode
episode_path = f"data/train/dress_the_teddy_bear/{Episode_Name}/episode.mcap"

local_file = hf_hub_download(
    repo_id=SOURCE_REPO,
    repo_type="dataset",
    filename=episode_path,
    local_dir=TEMP_DIR
)

with open(local_file, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    for channel_id, channel in summary.channels.items():
        count = summary.statistics.channel_message_counts.get(channel_id, 0)
        print(channel.topic, "->", count, "messages")