import torch
from collections import OrderedDict
from seamless_communication.inference import Translator

# Initialize a Translator object with a multitask model, vocoder on the GPU.
# translator = Translator(model_name_or_card="seamlessM4T_v2_large", vocoder_name_or_card='vocoder_36langs', device=torch.device("cuda:0"), dtype=torch.float16, input_modality='speech', output_modality='text')
translator = Translator(model_name_or_card="seamlessM4T_medium", vocoder_name_or_card=None, device=torch.device("cuda:0"), dtype=torch.float16, input_modality='speech', output_modality='text')

print()

translator.model.load_state_dict(torch.load("/models/m4t_finetuned/updated_state_dict.pt"))

print(type(translator))

text_output, _ = translator.predict(
    input='/datasets/test_3.wav', 
    task_str="S2TT",
    tgt_lang='kor'
)

# print(text_output[0])
print(str(text_output[0]))
print(type(str(text_output[0]))) # cstring to string