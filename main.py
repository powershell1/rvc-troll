import asyncio
from vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from config import (
    is_half,
    device
)
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
import edge_tts
import torch
import librosa
import soundfile as sf

TEXT = "Hello World!"
VOICE = "en-US-AnaNeural"
OUTPUT_FILE = "test.mp3"

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

cpt = torch.load(f"model/ayaka-jp.pth", map_location="cpu")
tgt_sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
if_f0 = cpt.get("f0", 1)
version = cpt.get("version", "v1")
if if_f0 == 1:
    net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
else:
    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
del net_g.enc_q
# print(net_g.load_state_dict(cpt["weight"], strict=False))
net_g.eval().to(device)
if is_half:
    net_g = net_g.half()
else:
    net_g = net_g.float()
vc = VC(tgt_sr, device, is_half)

async def amain() -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])

if __name__ == "__main__":
    load_hubert()
    """
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(amain())
    finally:
        loop.close()
    """

audio, sr = librosa.load("test.mp3", sr=16000, mono=True)
times = [0, 0, 0]
transpose = 0
method = "pm"
file_index = f"model/added_IVF1830_Flat_nprobe_9.index"
file_big_npy = f"model/total_fea.npy"
index_rate = 0.2

audio_opt = vc.pipeline(
    hubert_model,
    net_g,
    0,
    audio,
    times,
    transpose,
    method,
    file_index,
    file_big_npy,
    index_rate,
    if_f0,
)
# librosa.output.write_wav('file_trim_5s.wav', audio_opt, sr)
print(audio_opt)
sf.write('stereo_file.wav', audio_opt, tgt_sr, subtype='PCM_24')