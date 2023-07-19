import whisper

model = whisper.load_model("base.en")
result = model.transcribe("C:/Users/LHaiHui/Desktop/grounding_sam/speech/demo/speech.mp3")
print(result["text"])
