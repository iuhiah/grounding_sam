import whisper

model = whisper.load_model("base.en")
result = model.transcribe("") # path to audio file
print(result["text"])