import torch
from wrapper import PengiWrapper as Pengi
import os
import pandas as pd
import json
from tqdm import trange

DATA_BASE_DIR = 'D:\\\\PengiData'

def run_experiment(EXPERIMENT):
    pengi = Pengi(config='base', use_cuda=True)
    # Sound Event Classification
    ## ESC50
    if EXPERIMENT == 'ESC50':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'ESC50')
        csv_data = pd.read_csv(os.path.join(dataset_dir, 'meta.csv'))
        audio_file_names = csv_data['filename'].to_list()
        answers = csv_data['category'].to_list()
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    ## UrbanSound8K
    elif EXPERIMENT == 'UrbanSound8K':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'UrbanSound8K')
        csv_data = pd.read_csv(os.path.join(dataset_dir, 'UrbanSound8K.csv'))
        audio_file_names = csv_data['slice_file_name'].to_list()
        fold_id = csv_data['fold'].to_list()
        answers = csv_data['class'].to_list()
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', 'fold' + str(fold_id[i]), audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    ## DCASE2017Task4
    elif EXPERIMENT == 'DCASE2017Task4':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'DCASE2017Task4', 'test')
        audio_file_names = []
        a = []
        b = []
        answers = []
        with open(os.path.join(dataset_dir, 'meta.txt'), "r", encoding="utf-8") as file:
            for line in file:
                audio_file_names.append(line.split(',')[0].strip())
                a.append(line.split(',')[1].strip())
                b.append(line.split(',')[2].strip())
                ans = line.replace('\"', '')
                tmp_ans = ans.split(',')
                tmp_n = (len(tmp_ans) - 3) // 2
                answers.append(','.join(tmp_ans[3:3 + tmp_n]))
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', 'Y' + audio_file_names[i] + '_' + a[i] + '_' + b[i] + '.wav'))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    # Acoustic Scene Classification
    ## TUT2017
    elif EXPERIMENT == 'TUT2017':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'TUT2017', 'test')
        audio_file_names = []
        answers = []
        with open(os.path.join(dataset_dir, 'meta.txt'), "r", encoding="utf-8") as file:
            for line in file:
                audio_file_names.append(line.strip().split()[0])
                answers.append(line.strip().split()[1])
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    # Music
    ## Music Speech
    elif EXPERIMENT == 'MusicSpeech':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'GT.MusicSpeech')
        audio_file_names = os.listdir(os.path.join(dataset_dir, 'music')) + os.listdir(os.path.join(dataset_dir, 'speech'))
        for i in range(2):
            for j in range(64):
                if i == 0:
                    audio_file_names[i * 64 + j] = os.path.join('music', audio_file_names[i * 64 + j])
                if i == 1:
                    audio_file_names[i * 64 + j] = os.path.join('speech', audio_file_names[i * 64 + j])
        answers = ['music'] * 64 + ['speech'] * 64
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    ## Music Genres
    elif EXPERIMENT == 'MusicGenres':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'GT.MusicGenre', 'audio')
        audio_file_names = os.listdir(dataset_dir)
        answers = []
        for name in audio_file_names:
            answers.append(name.strip().split('.')[0].strip())
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    # Instrument Classification
    ## Beijing Opera
    elif EXPERIMENT == 'BeijingOpera':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'BeijingOpera')
        csv_data = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
        audio_file_names = csv_data['path'].to_list()
        answers = csv_data['classname'].to_list()
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    ## NSInstruments
    elif EXPERIMENT == 'NSInstruments':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'NSynth', 'test')
        audio_file_names = []
        answers = []
        with open(os.path.join(dataset_dir, 'meta.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)
            audio_file_names = list(data.keys())
            for i in range(len(audio_file_names)):
                answers.append(data[audio_file_names[i]]['instrument_family_str'])
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', audio_file_names[i] + '.wav'))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    # Emotion Recognition
    ## CREMA-D
    elif EXPERIMENT == 'CREMA-D':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'CREMA-D')
        audio_file_names = os.listdir(os.path.join(dataset_dir, 'audio'))
        answers = []
        for name in audio_file_names:
            ans = name[:-4].split('_')[-2]
            if ans == 'ANG':
                answers.append('anger')
            elif ans == 'DIS':
                answers.append('disgust')
            elif ans == 'FEA':
                answers.append('fear')
            elif ans == 'HAP':
                answers.append('happy')
            elif ans == 'NEU':
                answers.append('neutral')
            elif ans == 'SAD':
                answers.append('sad')
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
        #for i in range(100):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    ## RAVDESS
    elif EXPERIMENT == 'RAVDESS':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'RAVDESS')
        audio_file_names = os.listdir(os.path.join(dataset_dir, 'audio'))
        answers = []
        for name in audio_file_names:
            tmp_num = name.strip().split('-')[2].strip()
            if tmp_num == '01':
                answers.append('neutral')
            elif tmp_num == '02':
                answers.append('calm')
            elif tmp_num == '03':
                answers.append('happy')
            elif tmp_num == '04':
                answers.append('sad')
            elif tmp_num == '05':
                answers.append('angry')
            elif tmp_num == '06':
                answers.append('fearful')
            elif tmp_num == '07':
                answers.append('disgust')
            elif tmp_num == '08':
                answers.append('surprised')
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('') 
    # Vocal Sound Classification
    ## VocalSound
    elif EXPERIMENT == 'VocalSound':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'VocalSound')
        test_file_names = []
        with open(os.path.join(dataset_dir, 'test.txt'), "r", encoding="utf-8") as file:
            for line in file:
                test_file_names.append(line.strip().split(',')[0].strip())
        audio_file_names = []
        answers = []
        for file_name in os.listdir(os.path.join(dataset_dir, 'audio')):
            if file_name[:5] in test_file_names:
                audio_file_names.append(file_name)
                answers.append(file_name[:-4].split('_')[-1].strip())
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, 'audio', audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')
    # Surveillance
    ## SESA
    elif EXPERIMENT == 'SESA':
        dataset_dir = os.path.join(DATA_BASE_DIR, 'SESA', 'test')
        audio_file_names = os.listdir(dataset_dir)
        answers = []
        for name in audio_file_names:
            answers.append(name.split('_')[0])
        audio_file_paths = []
        text_prompts = []
        add_texts = []
        for i in range(len(audio_file_names)):
            audio_file_paths.append(os.path.join(dataset_dir, audio_file_names[i]))
            text_prompts.append('generate audio caption')
            add_texts.append('')

    # print(len(audio_file_names), len(answers))
    # print(audio_file_names[0])
    # print(answers[0])

    if EXPERIMENT == 'DCASE2017Task4':
        cats = [
            'Reversing beeps',
            'Bicycle',
            'Fire engine, fire truck (siren)',
            'Car',
            'Police car (siren)',
            'Skateboard',
            'Civil defense siren',
            'Train',
            'Car passing by',
            'Train horn',
            'Car alarm',
            'Motorcycle',
            'Bus',
            'Truck',
            'Air horn, truck horn',
            'Screaming',
            'Ambulance (siren)',
        ]
        idx2cat = {}
        for i in range(len(cats)):
            idx2cat[i] = cats[i]
    else:
        idx2cat = {}
        flag = 0
        for i in range(len(answers)):
            if answers[i] not in idx2cat.values():
                idx2cat[flag] = answers[i]
                flag += 1
    # print(idx2cat)
    # print(len(idx2cat))
    cat_embeddings = torch.zeros((len(idx2cat), 1024), dtype=torch.float)
    for i in range(len(idx2cat)):
        tmp_embedding = pengi.get_prompt_embeddings(prompts=[idx2cat[i]])[1].squeeze().cpu()
        cat_embeddings[i] = tmp_embedding
    cat_embeddings = cat_embeddings / torch.norm(cat_embeddings, dim=1, keepdim=True)

    N = 64
    generated_response = []
    for i in trange(len(audio_file_paths) // N + int((len(audio_file_paths) % N) > 0)):
        tmp_audio_file_paths = audio_file_paths[i * N:(i + 1) * N]
        tmp_text_prompts = text_prompts[i * N:(i + 1) * N]
        tmp_add_texts = add_texts[i * N:(i + 1) * N]
        tmp_generated_response = pengi.generate(
                                                audio_paths=tmp_audio_file_paths,
                                                text_prompts=tmp_text_prompts, 
                                                add_texts=tmp_add_texts, 
                                                max_len=30, 
                                                beam_size=5, 
                                                temperature=1.0, 
                                                stop_token=' <|endoftext|>',
                                                )
        for j in range(len(tmp_generated_response)):
            index = torch.argmax(tmp_generated_response[j][1])
            tmp_result = tmp_generated_response[j][0][index]
            max_score = -2
            tmp_idx = -1
            tmp_embedding = pengi.get_prompt_embeddings(prompts=[tmp_result])[1].squeeze().cpu()
            tmp_embedding = tmp_embedding / torch.norm(tmp_embedding)
            tmp_scores = torch.sum(cat_embeddings * tmp_embedding.unsqueeze(0), dim=1)
            if torch.max(tmp_scores).item() > max_score:
                tmp_idx = torch.argmax(tmp_scores).item()
                max_score = torch.max(tmp_scores).item()
            #print(tmp_result, ':,', idx2cat[tmp_idx])
            if tmp_idx < 0:
                print('ERROR')
            generated_response.append(idx2cat[tmp_idx])

    with open(EXPERIMENT + "_Result_beam5.txt", "w") as file:
        for item in generated_response:
            file.write(str(item) + "\n")

    all_v = 0
    all_w = 0
    compare = []
    for i in range(len(generated_response)):
        gt_cat = answers[i]
        all_v += float(generated_response[i] in gt_cat)
        all_w += 1
        compare.append(f'{gt_cat:^40}:{generated_response[i]:^40}')

    with open(EXPERIMENT + "_Compare_beam5.txt", 'w') as file:
        for item in compare:
            file.write(item + '\n')

    # print(all_v / all_w)
    with open(EXPERIMENT + "_Accuracy_beam5.txt", 'w') as file:
        file.write(str(all_v / all_w) + '\n')

EXPERIMENTs = [
    'ESC50',
    'UrbanSound8K',
    'DCASE2017Task4',
    'TUT2017',
    'MusicSpeech',
    'MusicGenres',
    'BeijingOpera',
    'NSInstruments',
    'CREMA-D',
    'RAVDESS',
    'VocalSound',
    'SESA'
]

if __name__ == '__main__':
    # Run All Experiments
    for EXPERIMENT in EXPERIMENTs:
        run_experiment(EXPERIMENT)