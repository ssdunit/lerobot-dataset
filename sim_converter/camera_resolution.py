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
from mcap_protobuf.decoder import DecoderFactory

with open(local_file, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    seen = {"/instruction": 0, "/top-camera": 0, "/left-arm-action": 0, "/left-arm-state": 0}
    for schema, channel, message, decoded in reader.iter_decoded_messages():
        if channel.topic in seen and seen[channel.topic] < 1:
            print(f"--- {channel.topic} ---")
            print(decoded)
            print()
            seen[channel.topic] += 1
        if all(v >= 1 for v in seen.values()):
            break