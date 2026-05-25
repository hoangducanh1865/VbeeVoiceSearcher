from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader

# EspeakWrapper.set_library(espeakng_loader.get_library_path())
# EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

# phonemes = phonemize('Hello')
# print('Phonemes: ', phonemes)

import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)