import nlpaug.augmenter.word as naw
from scipy.linalg import get_blas_funcs, triu

augmentation = naw.WordEmbsAug(
	model_type="glove",
	model_path="glove.6B.100d.txt",
	action="substitute"
)

text_original = "The pen is mightier than the sword."
text_aug = augmentation.augment(text_original)
print(text_aug)
