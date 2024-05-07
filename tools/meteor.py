import sys
import nltk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')

if __name__ == "__main__":
    pred_path = sys.argv[1]
    data_path = sys.argv[2]
    with open(pred_path, "r") as file:
        pred = file.readlines()

    with open(data_path, "r") as file:
        target = file.readlines()

    # Chia các câu dữ liệu dự đoán và câu tham chiếu thành danh sách các từ (pre-tokenized hypothesis)
    pred_tokenized = [word_tokenize(p.lower()) for p in pred]
    target_tokenized = [word_tokenize(t.lower()) for t in target]

    # Tính toán METEOR
    scores = [nltk.meteor([t], p) for t, p in zip(target_tokenized, pred_tokenized)]
    average_score = sum(scores) / len(scores)
    print(average_score)
