import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# 1. 모델 설정 로드
model_dir = "path/to/your/model/directory"  # 모델 파일이 저장된 경로로 설정

# config.json 로드
config = AutoConfig.from_pretrained(model_dir)

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 2. 모델 가중치 로드
# .pth 파일을 로드하여 모델 초기화
model = AutoModelForCausalLM.from_config(config)

# 모델 가중치 로드
model.load_state_dict(torch.load(f"{model_dir}/consolidated.00.pth", map_location=torch.device("cpu")))  # CPU로 로드 (GPU로 변경 가능)

# 3. 모델 테스트 예시
input_text = "Test text to generate"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)

# 생성된 텍스트 출력
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
