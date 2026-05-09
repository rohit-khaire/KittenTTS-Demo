from kittentts.onnx_model import KittenTTS_1_Onnx
import soundfile as sf

# Saves as Output.wav file

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
This Kitten TTS model is now running fully offline.
Perfect for hackathons and competitions.
"""

# Generate speech
audio = model.generate(
    text=text,
    voice=voice
)

# Save output
sf.write("output.wav", audio, 24000)

print("Offline audio generated successfully!")