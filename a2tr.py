from utils import *
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("../all-mpnet-base-v2")

file_path = "../clotho/clotho_captions_evaluation.csv"
# audio_path = "../clotho/evaluation"
# out_path = "../clotho/evaluation_trimmed"
ks = [1, 5, 10]

audio, captions = load_csv_data(file_path)

# 生成音频描述
# audio_captions = get_audio_captions(audio)
with open("./audio_captions_beam5_size1.txt", "r") as f:
    audio_captions = [line.strip() for line in f]
print(audio_captions[0])

# 计算文本嵌入
# text_embeddings = get_text_embedding_pengi(captions)
# torch.save(text_embeddings, "text_embeddings.pt")
# text_embeddings = torch.load("text_embeddings.pt")
text_embeddings = torch.tensor(model.encode(captions))

# audio_captions_embeddings = get_text_embedding_pengi(audio_captions)
# torch.save(audio_captions_embeddings, "audio_captions_embeddings_beam5_size1.pt")
# audio_captions_embeddings = torch.load("audio_captions_embeddings_beam5_size1.pt")
# audio_captions_embeddings = audio_captions_embeddings[0]
audio_captions_embeddings = torch.tensor(model.encode(audio_captions))

# 计算 Recall @ ks
# recall_atr_ks = compute_recall_at_k(audio_captions_embeddings, text_embeddings, ks)
RPG(audio_captions_embeddings, text_embeddings, ks, audio_captions, captions, batch_size=4, candidate_num=20)

# SPIDEr = calculate_spider(hypotheses=audio_captions, references=captions)
# print(SPIDEr)