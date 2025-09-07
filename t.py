import torch

if torch.cuda.is_available():
    print("CUDA доступна ✅")
    print("Количество устройств:", torch.cuda.device_count())
    print("Имя GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA недоступна ❌")
