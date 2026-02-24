"""
Qwen3-TTS 4-bit Dequant Sample

Loads the pre-quantized 4-bit model (~486MB download) and generates speech.
Optimized for GPUs without native bf16 support (e.g. GTX 1080).

Requirements:
    pip install -e .

Usage:
    python sample.py
"""
from qwen_tts import load_4bit_dequant
import soundfile as sf

model = load_4bit_dequant()

wavs, sr = model.generate_custom_voice(
    text="가장 아름다운 시간은 당신과 함께 보내는 지금.",
    language="Korean",
    speaker="Sohee",
    non_streaming_mode=False,
)

sf.write("output.wav", wavs[0], sr)
print(f"Saved output.wav ({len(wavs[0])/sr:.2f}s, {sr}Hz)")

# Available speakers: Aiden, Dylan, Eric, Ono_Anna, Ryan, Serena, Sohee, Uncle_Fu, Vivian
# Available languages: Chinese, English, Japanese, Korean, German, French, Russian,
#                      Portuguese, Spanish, Italian, Auto
