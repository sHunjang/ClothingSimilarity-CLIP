import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

#clip.load(): OpenAI에서 학습된 CLIP 모델 전처리와 전처리 함수 Load
model, preprocess = clip.load("ViT-B/32", device=device)    # ViT-B/32 : Vision Transformer 기반 CLIP 모델


# 이미지를 모델에 넣기 전에 전처리 과정 함수 (224x224로 사이즈 조정, 정규화 등)
def get_image_feature(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)     # encode_image(): CLIP이 이미지 -> 벡터(512차원)로 변환
        feature = feature / feature.norm(dim=-1, keepdim=True)      # Cosine Similarity를 위해 정규화(norm=1)
    return feature.to("cpu")    # 이후 연산의 안정성을 위해 "CPU"로 이동


# Dataset 하위 폴더들(T-shirt, Hoodie, Jeans, ..)을 클래스(Class)로 간주
def load_dataset_features(dataset_root):
    features = []   # 임베딩 벡터
    labels = []     # 클래스명 (ex. hoodie, jeans, ..)
    paths = []      # 실제 파일 경로
    for class_folder in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_folder)
        if not os.path.isdir(class_path): continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.png')):
                path = os.path.join(class_path, file)
                feature = get_image_feature(path)
                features.append(feature)
                labels.append(class_folder)  # 클래스명 저장, 정답 라벨
                paths.append(path)
    return torch.cat(features), labels, paths   # torch.cat(features) 로 벡터들을 [N, 512] 텐서로 묶어 반환


# Query 벡터와 모든 데이터셋 이미지 벡터 간 Cosine Similarity 계산
def get_topk_similar(query_feature, dataset_features, labels, paths, k=5):

    scores = (dataset_features @ query_feature.T).squeeze(1)
    # (dataset_features @ query_feature.T): 벡터 내적 = Cosine Simialrity -> 벡터 정규화 했기 때문

    topk = scores.topk(k).indices.tolist()  # topk(): 유사도 높은 순서대로 K개 추출
    return [(paths[i], labels[i], scores[i].item()) for i in topk]  # [(경로, 클래스, 유사도)] 순으로 출력됨.


# 쿼리 이미지 파일명을 기준으로 정답 class 추출 (ex. Hoodie_01.jpg -> hoodie)
# Top-K 결과 중에서 정답 Class가 포함되어 있는지 확인
# 전체 쿼리 이미지들에 대해 Accuracy 계산
def evaluate(query_folder, dataset_root, k=5):
    dataset_features, dataset_labels, dataset_paths = load_dataset_features(dataset_root)
    correct = 0
    total = 0

    for file in tqdm(os.listdir(query_folder)):
        if not file.lower().endswith(('.jpg', '.png')): continue
        query_path = os.path.join(query_folder, file)
        query_feature = get_image_feature(query_path)

        # 정답 클래스명 추출 ex: "hoodie_01.jpg"
        gt_class = file.split("_")[0]  # ex: hoodie

        topk = get_topk_similar(query_feature, dataset_features, dataset_labels, dataset_paths, k)

        predicted_classes = [label for _, label, _ in topk]
        if gt_class in predicted_classes:
            correct += 1
        total += 1

        # 결과 출력
        print(f"\n🔍 Query: {file} | 정답: {gt_class}")
        for i, (path, label, score) in enumerate(topk):
            print(f"  {i+1}. {label} ({score:.4f}) - {os.path.basename(path)}")

    acc = correct / total if total else 0
    print(f"\n✅ Top-{k} Accuracy: {acc:.2%} ({correct}/{total})")



if __name__ == "__main__":
    # 상의 기준 실험
    evaluate("queries/tops", "dataset/tops", k=5)