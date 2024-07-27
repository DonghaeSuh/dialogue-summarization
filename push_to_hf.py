from huggingface_hub import HfApi
from huggingface_hub.hf_api import HfFolder
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv('HF_API_KEY')
HfFolder.save_token(HF_API_KEY)

api = HfApi()

api.upload_folder(
    folder_path="/content/drive/MyDrive/국립국어원_일상대화요약/korean_dialog/korean_dialog/run/model/MLP-KTLim/llama-3-Korean-Bllossom-8B_batch_1_preprocess7_time_2024-07-21_22:21",
    repo_id="ywhwang/llama-3-Korean-Bllossom-8B",
    repo_type="model",
)