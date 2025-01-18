import torch
import torch.nn.functional as F
from wrapper import PengiWrapper as Pengi
import pandas as pd
from tqdm import tqdm, trange
from pydub import AudioSegment
import os
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from torch.cuda.amp import autocast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

# captions_embeddings: 模型为数据集生成的音频描述嵌入 (N, D)
# groundtruth_embeddings: 测试样例的 Groundtruth 嵌入 (M, D)
def comput_cosine_similarity(captions_embeddings, groundtruth_embeddings):
    # 确保数据在 GPU 上
    captions_embeddings = captions_embeddings.to("cuda")
    groundtruth_embeddings = groundtruth_embeddings.to("cuda")

    # 归一化嵌入向量，方便后续计算余弦相似度
    captions_embeddings = F.normalize(captions_embeddings, p=2, dim=1, eps=1e-8)
    groundtruth_embeddings = F.normalize(groundtruth_embeddings, p=2, dim=1, eps=1e-8)

    # 计算余弦相似度 (M, N): groundtruth_embeddings 与 captions_embeddings 的矩阵乘法
    similarity_matrix = torch.matmul(groundtruth_embeddings, captions_embeddings.T)

    return similarity_matrix

def get_top_k_indices(captions_embeddings, groundtruth_embeddings, k):
    similarity_matrix = comput_cosine_similarity(captions_embeddings, groundtruth_embeddings)

    # 计算 Recall@k
    top_k_indices = torch.topk(similarity_matrix, k, dim=1).indices  # (M, k)
    # print(top_k_indices)
    top_k_indices_list = top_k_indices.squeeze().tolist()
    # print(top_k_indices_list)
    return top_k_indices_list

def compute_recall_at_k(captions_embeddings, groundtruth_embeddings, ks):
    """
    计算 R@k
    :param captions_embeddings: Tensor (N, D)，数据集中音频描述的嵌入
    :param groundtruth_embeddings: Tensor (M, D)，测试样例的 Groundtruth 嵌入
    :param ks: list, Top-K 的值
    :return: recall_at_k: int, R@k 的值
    """
    similarity_matrix = comput_cosine_similarity(captions_embeddings, groundtruth_embeddings)

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


def rethinking(model, tokenizer, top_k_indices, generated_captions, groundtruth_caption):
    device = "cuda"

    # 将模型移动到 GPU 上
    model = model.to(device)

    top_k_captions = [[i, generated_captions[i]] for i in top_k_indices]
    valid_answers = []

    for index, caption in top_k_captions:
        prompt = f"Do the following captions desribing the same audio?\nA. {caption}\nB. {groundtruth_caption}\nAnswer with ONLY 'YES' or 'NO' DIRECTLY."
        messages = [
                    {"role": "system", "content": "You are Qwen, a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
        # input_ids = inputs.input_ids.to(device)
        # attention_mask = inputs.attention_mask.to(device)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if "YES" in generated_text:
            valid_answers.append(index)

    while len(valid_answers) <= 20:
        valid_answers.append(-1)
    return valid_answers

def recall_at_k(top_k_indices, groundtruth_embeddings, k):
    # 构造 Groundtruth 索引
    groundtruth_indices = []
    for i in range(len(groundtruth_embeddings)):
        groundtruth_indices.append(i // 5)
    groundtruth_indices = torch.tensor(groundtruth_indices, device="cuda")
    # 计算 Recall@k
    recall_at_ks = []
    top_k_indices = [row[:k] for row in top_k_indices]
    top_k_indices = torch.tensor(top_k_indices, device="cuda")

    # print(top_k_indices.shape)
    # print(groundtruth_indices.shape)
    # 对每个 groundtruth_caption，取 top_k 相似的索引
    matches = (top_k_indices == groundtruth_indices.unsqueeze(1)).any(dim=1).float()
    recall_at_k = matches.mean().item()
    print(f"R@{k}: {recall_at_k}")
    recall_at_ks.append(recall_at_k)

def RPG(captions_embeddings, groundtruth_embeddings, ks, generated_captions, groundtruth_captions, batch_size=4, candidate_num=30):
    # 加载预训练的 GPT-2 分词器和模型
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    model.eval()

    # 获取 top 20 索引
    top_20_indices = get_top_k_indices(captions_embeddings, groundtruth_embeddings, candidate_num)
    top_k_indices = []

    # 批量处理数据
    batch_size = batch_size  # 可以根据 GPU 内存调整批次大小
    total_len = 0
    with torch.no_grad(), autocast():
        for i in trange(0, len(top_20_indices), batch_size):
            batch_indices = top_20_indices[i:i + batch_size]
            batch_groundtruth_captions = groundtruth_captions[i:i + batch_size]

            # 构造批量输入
            prompts = []
            for indices, groundtruth_caption in zip(batch_indices, batch_groundtruth_captions):
                for index in indices:
                    caption = generated_captions[index]
                    prompt = f"Are the following captions likely to be describing the same audio?\nA. {caption}\nB. {groundtruth_caption}\nAnswer with ONLY 'YES' or 'NO' DIRECTLY."
                    prompts.append(prompt)

            # 批量生成回答
            texts = [tokenizer.apply_chat_template([{"role": "system", "content": "You are Qwen, a helpful assistant."}, {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")


            generated_ids = model.generate(**model_inputs, max_new_tokens=10, temperature=0.1)  # 减少 max_new_tokens 以加速推理

            # 解码生成的文本
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 处理生成的文本
            valid_answers = []
            for j in range(0, len(generated_texts), candidate_num):
                batch_texts = generated_texts[j:j + candidate_num]  # 当前批次的生成文本
                current_batch_indices = batch_indices[j // candidate_num]  # 对应的索引批次

                # 提取每个文本的 "YES" 或 "NO"
                valid_indices = []
                for idx, text in zip(current_batch_indices, batch_texts):
                    # 提取 assistant 的回答
                    assistant_response = text.split("assistant")[-1].strip()
                    # print(assistant_response)
                    if "YES" in assistant_response:  # 如果回答是 "YES"
                        valid_indices.append(idx)  # 记录有效索引

                # 统计有效索引数量
                total_len += len(valid_indices)

                # 如果 valid_indices 不足 20 个，用 -1 填充
                while len(valid_indices) < 20:
                    valid_indices.append(-1)

                # 将当前批次的 valid_indices 加入 valid_answers
                valid_answers.append(valid_indices)

            top_k_indices.extend(valid_answers)
    print(total_len / len(groundtruth_captions))

    # 计算 Recall@k
    for k in ks:
        recall_at_k(top_k_indices, groundtruth_embeddings, k)


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


def calculate_spider(hypotheses, references):
    """
    Calculate SPIDEr score for audio captioning tasks.

    :param hypotheses: Dict mapping audio IDs to generated captions.
    :param references: Dict mapping audio IDs to reference captions.
    :return: Average SPIDEr score.
    """
    # Prepare data in COCO format
    gts = {}
    res = {}

    N = len(hypotheses)
    if len(references) != 5 * N:
        raise ValueError("参考总数必须是预测数的 5 倍！")

    for i in range(N):
        # 第 i 个样本的 1 条预测
        res[i] = [hypotheses[i]]
        # 第 i 个样本的 5 条参考
        gts[i] = references[5*i : 5*i + 5]

    # 计算 CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    # 计算 SPICE
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts, res)

    # 计算 SPIDEr
    spider_score = (cider_score + spice_score) / 2.0
    return spider_score
