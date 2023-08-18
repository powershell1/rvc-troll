import asyncio
from vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import edge_tts
import torch
import librosa
import scipy
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

voice = "en-US-JennyNeural"
output_file = "IdolVocal.mp3"

config = Config()

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def load_model(pth_file):
    cpt = torch.load(pth_file, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    """
    if if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    """
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    return vc, net_g, if_f0, tgt_sr, version

def run_vc(vc, net_g, audio, if_f0, tgt_sr, version):
    times = [0, 0, 0]
    transpose = 0
    method = "harvest"
    index_rate = 0.2
    filter_radius = 3
    resample = 0
    protect = 0.5
    audio_opt = vc.pipeline(
        model=hubert_model,
        net_g=net_g,
        sid=0,
        audio=audio,
        input_audio_path=output_file,
        times=times,

        f0_up_key=transpose,
        f0_method=method,

        file_index=file_index,
        # file_big_npy,
        index_rate=index_rate,
        if_f0=if_f0,

        filter_radius=filter_radius,
        tgt_sr=tgt_sr,
        resample_sr=resample,
        rms_mix_rate=0,
        version=version,
        protect=protect,
        f0_file=None,
    )
    return audio_opt


async def processingTTS(text) -> None:
    communicate = edge_tts.Communicate(text, voice)
    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])

math_folder = f"model/viper"
file_pth = math_folder+f"/viper.pth"
file_index = math_folder+f"/added_IVF924_Flat_nprobe_1_viper_v2.index"
# file_big_npy = f"model/total_fea.npy"
vc, net_g, if_f0, tgt_sr, version = load_model(file_pth)

#p = pyaudio.PyAudio()
#stream = p.open(format=pyaudio.paInt16, channels=1, rate=tgt_sr, output=True, output_device_index=5)

def inputLoop():
    input_text = input("Enter a word: ")
    asyncio.run(processingTTS(input_text))
    audio, sr = librosa.load(output_file, sr=16000, mono=True)
    audio_run = run_vc(vc, net_g, audio, if_f0, tgt_sr, version)
    # scipy.io.wavfile.write('stereo_file2.wav', tgt_sr, audio_run)
    # stream.write(audio_run.tobytes())
    inputLoop()

"""
stream.stop_stream()
stream.close()
p.terminate()
"""

if __name__ == "__main__":
    load_hubert()
    audio, sr = librosa.load(output_file, sr=16000, mono=True)
    audio_run = run_vc(vc, net_g, audio, if_f0, tgt_sr, version)
    scipy.io.wavfile.write('stereo_file2.wav', tgt_sr, audio_run)
    #inputLoop()
    """
    audio_run = audio_run.astype(numpy.uint16)
    hex_table = numpy.vectorize(hex)(audio_run)
    print(hex_table)
    """

#scipy.io.wavfile.write('stereo_file2.wav', tgt_sr, audio_opt)