from kittentts.onnx_model import KittenTTS_1_Onnx
import sounddevice as sd

# Load local offline ONNX model
model = KittenTTS_1_Onnx(
    model_path="models/kitten_tts_mini_v0_8.onnx",
    voices_path="models/voices.npz"
)

# Available voices:
# expr-voice-2-m
# expr-voice-2-f
# expr-voice-3-m
# expr-voice-3-f
# expr-voice-4-m
# expr-voice-4-f
# expr-voice-5-m
# expr-voice-5-f

voice = "expr-voice-2-m"

text = """
Hello Rohit.
This Kitten TTS model is speaking directly
without saving audio files.
"""

# Generate audio
audio = model.generate(
    text=text,
    voice=voice
)

# Play audio directly
sd.play(audio, samplerate=24000)

# Wait until playback finishes
sd.wait()

print("Audio playback completed!")