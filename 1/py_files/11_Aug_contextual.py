import nlpaug.augmenter.word as naw
import torch

device = "cuda" if torch.cuda.is_available() else "cpu" 
print(device)

model_name = "bert-base-uncased"
augmentation = naw.ContextualWordEmbsAug(model_path=model_name,
										 action="substitute",
										 device="cpu")
text_original = "The pen is mightier than the sword."
text_aug = augmentation.augment(text_original)

print(text_aug)
