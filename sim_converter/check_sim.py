from huggingface_hub import hf_hub_download
from mcap.reader import make_reader

SOURCE_REPO = "XDOF/ABC-130k"
episode_path = "data/sim/pouring_beads/episode_019e4b26-47fe-72bb-8893-71a48aee3268/episode.mcap"

local_file = hf_hub_download(
    repo_id=SOURCE_REPO,
    repo_type="dataset",
    filename=episode_path,
    local_dir="./temp_mcap_sim"
)

with open(local_file, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    for channel_id, channel in summary.channels.items():
        count = summary.statistics.channel_message_counts.get(channel_id, 0)
        print(channel.topic, "->", count, "messages")