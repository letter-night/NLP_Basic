from transformers import pipeline

pipe = pipeline("fill-mask")
new_text = pipe("The pen is <mask> than the sword.")

print(new_text)
