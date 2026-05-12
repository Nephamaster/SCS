import json

def merge_conversations(item):
    """
    将一个条目的 human/gpt 对话按顺序拼接成一段文本
    """
    conversations = item.get("conversations", [])
    texts = []
    for turn in conversations:
        value = turn.get("value", "").strip()
        texts.append(value)
    return " ".join(texts)

def convert_file_to_text_list(input_path, output_path=None):
    """
    将 JSON 文件转为列表，每个元素是一条拼接文本
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_list = [merge_conversations(item) for item in data]

    if output_path:
        # 保存为 JSON 列表
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(text_list, f, ensure_ascii=False, indent=2)

    return text_list

if __name__ == "__main__":
    input_path = "../../data/SFT.json"
    output_path = "../../data/SFT_text.json"
    texts = convert_file_to_text_list(input_path, output_path)
    print(texts[:3])  # 打印前3条样例查看