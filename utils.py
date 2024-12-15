import torch
import torch.nn.functional as F
from wrapper import PengiWrapper as Pengi
import pandas as pd
from pydub import AudioSegment
import os


# captions_embeddings: 模型为数据集生成的音频描述嵌入 (N, D)
# groundtruth_embeddings: 测试样例的 Groundtruth 嵌入 (M, D)

def compute_recall_at_k(captions_embeddings, groundtruth_embeddings, ks):
    """
    计算 R@k
    :param captions_embeddings: Tensor (N, D)，数据集中音频描述的嵌入
    :param groundtruth_embeddings: Tensor (M, D)，测试样例的 Groundtruth 嵌入
    :param ks: list, Top-K 的值
    :return: recall_at_k: int, R@k 的值
    """

    # 确保数据在 GPU 上
    captions_embeddings = captions_embeddings.to("cuda")
    groundtruth_embeddings = groundtruth_embeddings.to("cuda")

    # 归一化嵌入向量，方便后续计算余弦相似度
    captions_embeddings = F.normalize(captions_embeddings, p=2, dim=1, eps=1e-8)
    groundtruth_embeddings = F.normalize(groundtruth_embeddings, p=2, dim=1, eps=1e-8)

    # 计算余弦相似度 (M, N): groundtruth_embeddings 与 captions_embeddings 的矩阵乘法
    similarity_matrix = torch.matmul(groundtruth_embeddings, captions_embeddings.T)

    # 构造 Groundtruth 索引
    groundtruth_indices = []
    for i in range(len(groundtruth_embeddings)):
        groundtruth_indices.append(i // 5)
    groundtruth_indices = torch.tensor(groundtruth_indices, device="cuda")
    # print(groundtruth_indices[:10])

    # 计算 Recall@k
    recall_at_ks = []
    # 对每个 groundtruth_caption，取 top_k 相似的索引
    for k in ks:
        top_k_indices = torch.topk(similarity_matrix, k, dim=1).indices  # (M, k)
        matches = (top_k_indices == groundtruth_indices.unsqueeze(1)).any(dim=1).float()
        recall_at_k = matches.mean().item()
        print(f"R@{k}: {recall_at_k}")
        recall_at_ks.append(recall_at_k)

    return recall_at_ks


def get_text_embedding_pengi(text):
    """
    使用 Pengi 模型生成文本嵌入
    :param text: list, 对应的文本描述
    :return: text_embedding: Tensor (D, ), 文本嵌入
    """
    # 初始化 Pengi 模型
    pengi = Pengi(config="base")

    # 生成文本嵌入
    text_prefix, text_embeddings = pengi.get_prompt_embeddings(prompts=text)

    return text_embeddings

def get_audio_captions(audio):
    pengi = Pengi(config="base")
    prompts = ["generate audio caption"] * len(audio)
    add_texts = [""] * len(audio)

    generated_response = pengi.generate(audio_paths=audio,
                                        text_prompts=prompts,
                                        add_texts=add_texts,
                                        max_len=30,
                                        beam_size=5,
                                        temperature=1.0,
                                        stop_token=' <|endoftext|>'
                                        )

    captions = []

    for caption, score in generated_response:
        best_index = score.argmax().item()
        captions.append(caption[best_index])

    print(len(captions))

    with open("./audio_captions_beam5_size1.txt", "w") as f:
        for item in captions:
            f.write(item + "\n")

    return captions

def load_csv_data(filepath):
    """
    从 CSV 文件中加载数据
    :param filepath: str, CSV 文件路径
    :return: audio, 音频文件地址
            -captions, 音频描述
    """
    data = pd.read_csv(filepath)
    data["file_name"] = "../clotho/evaluation/" + data["file_name"]

    caption = data[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
    audio = data["file_name"].tolist()

    # print(audio[:10])
    # print(caption[:10])

    return audio, caption
