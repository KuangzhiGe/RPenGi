from wrapper import PengiWrapper as Pengi
from utils import *
import pandas as pd

#pengi = Pengi(config="base") #base or base_no_text_enc,

data = pd.read_csv('../clotho/clotho_captions_evaluation.csv')
caption = data[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.tolist()
audio = data["file_name"].tolist()
#print(len(audio), len(caption[0]))
pengi = Pengi(config="base")

audio_file_paths = ['../clotho/clotho_audio_evaluation/evaluation/' + audio[0]]
text_prompts = ['Is ' + '\"' + caption[0][0] + '\"' ' a correct caption for this audio? Answer \"Yes\" or \"No\"']
add_texts = [""]
'''
generated_response = pengi.generate(
                                    audio_paths=audio_file_paths,
                                    text_prompts=text_prompts, 
                                    add_texts=add_texts, 
                                    max_len=30, 
                                    beam_size=5, 
                                    temperature=1.0, 
                                    stop_token=' <|endoftext|>',
                                    )
print(generated_response)
'''
generated_summary = pengi.describe(
                                    audio_paths=audio_file_paths,
                                    max_len=30, 
                                    beam_size=5,  
                                    temperature=1.0,  
                                    stop_token=' <|endoftext|>',
                                    )
print(caption[0])
print(generated_summary)
'''
audio_prefix, audio_embeddings = pengi.get_audio_embeddings(audio_paths=audio_file_paths)

text_prefix, text_embeddings = pengi.get_prompt_embeddings(prompts=text_prompts)
print(audio_prefix, text_prefix)
'''