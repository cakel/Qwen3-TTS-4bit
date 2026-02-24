# Qwen3-TTS-4bit

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 0.6B CustomVoice 모델을 **4-bit 양자화 가중치**로 로드하여 소비자용 GPU에서 TTS를 실행하는 로더입니다.

## 왜 만들었나

원본 Qwen3-TTS 모델은 bf16으로 동작하지만, GTX 1080 같은 구형 GPU는 bf16을 네이티브 지원하지 않아 fp32로 변환해야 합니다. 이 경우 VRAM 사용량이 두 배로 늘어 8GB VRAM으로는 모델 로드 자체가 어렵고, 추론 속도도 느려집니다. 4-bit 양자화 가중치를 사용하면 다운로드 용량과 VRAM 사용량을 크게 줄여, 구형 GPU에서도 간단한 음성 생성이 가능합니다.

## 특징

- 다운로드: ~486MB ([Wookidooki/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit](https://huggingface.co/Wookidooki/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit))
- bitsandbytes 불필요
- Ampere 이전 GPU (GTX 1080 등) 자동 감지 → fp32 전환
- 첫 실행 시 dequantize 후 디스크 캐시 → 이후 빠른 로딩

## 설치

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

git clone https://github.com/cakel/Qwen3-TTS-4bit.git
cd Qwen3-TTS-4bit
pip install -e .
```

## 사용법

```python
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
```

## 지원 스피커

| Speaker | 설명 | 네이티브 언어 |
|---|---|---|
| Vivian | 밝고 날카로운 젊은 여성 | Chinese |
| Serena | 따뜻하고 부드러운 젊은 여성 | Chinese |
| Uncle_Fu | 낮고 부드러운 중년 남성 | Chinese |
| Dylan | 자연스럽고 맑은 베이징 남성 | Chinese (Beijing) |
| Eric | 활기찬 청두 남성 | Chinese (Sichuan) |
| Ryan | 리드미컬한 남성 | English |
| Aiden | 밝고 맑은 미국 남성 | English |
| Ono_Anna | 경쾌한 일본 여성 | Japanese |
| Sohee | 감성적인 한국 여성 | Korean |

## 지원 언어

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## GPU 호환성

| GPU | compute_dtype | 비고 |
|---|---|---|
| Ampere 이상 (RTX 30xx, 40xx) | bf16 | 네이티브 bf16 지원 |
| Pascal/Turing (GTX 10xx, RTX 20xx) | fp32 | 자동 감지, fp32 전환 |

## 제한 사항

이 저장소는 0.6B 4-bit 양자화 모델을 사용하므로, **긴 문장에서 발음이 부정확해지거나 억양이 부자연스러워질 수 있습니다.** 짧은 문장(1~2문장)에 적합하며, 긴 텍스트의 고품질 음성 생성이 필요한 경우 원본 저장소의 bf16 모델(1.7B 등)을 사용하세요.

## 원본 프로젝트

이 저장소는 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)의 fork입니다. 원본 모델 및 전체 기능(Voice Design, Voice Clone, Fine-tuning, 1.7B 모델 등)은 원본 저장소를 참조하세요.
