import torch
from collections import Counter

def check_dataset_sanity(dataset, expected_label_size=None, dataset_name="Dataset"):
    print(f"==== [Sanity Check for {dataset_name}] ====")

    labels = [int(label) for _, label in dataset]
    label_tensor = torch.tensor(labels)

    # 1. 样本数量 vs 标签数量
    print(f"Total samples: {len(dataset)}")
    print(f"Label tensor shape: {label_tensor.shape}")

    # 2. 标签是否是整数型
    if not torch.all(label_tensor == label_tensor.long()):
        print("ERROR: 标签中包含非整数类型！")

    # 3. 标签是否落在合法范围内
    min_label = int(label_tensor.min().item())
    max_label = int(label_tensor.max().item())
    print(f"Label value range: {min_label} ~ {max_label}")

    # 4. 标签类别统计
    label_counter = Counter(labels)
    print(f"Label counts: {dict(label_counter)}")

    # 5. 验证标签是否与模型输出维度匹配
    if expected_label_size is not None:
        if max_label >= expected_label_size:
            print(f"ERROR: 标签值超出模型类别数范围！")
            print(f"  标签最大值: {max_label}，但模型输出维度为: {expected_label_size}")
        else:
            print(f"标签值在模型输出范围之内。")

    print("==== Sanity Check Done ====\n")
