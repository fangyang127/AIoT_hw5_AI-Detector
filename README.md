# AI / Human 文章偵測器

使用 Hugging Face 模型（`roberta-base-openai-detector`）在本地快速判定文本為 **AI 生成** 或 **人類撰寫** 的機率。介面採用 Streamlit，支援文字輸入與檔案上傳。

Demo site：<https://aiot-hw5-ai-detector-7114056047.streamlit.app/>

## 功能
- 文字輸入框與 `.txt` 檔案上傳
- 即時顯示 AI% / Human%（進度條與文字結論）
- 信心不足時顯示「不確定」提示
- 內建中英樣例文本，方便測試
- 推論僅在本地進行，不上傳資料

## 安裝與執行
```bash
pip install -r requirements.txt
streamlit run app.py
```
預設使用 CPU，首次啟動會自動下載模型權重（需網路）。之後離線亦可使用。

## 使用方式
1. 在介面輸入文字，或上傳 `.txt` 純文字檔。
2. 按「開始偵測」取得 AI% / Human% 與模型結論。
3. 若提示信心偏低，表示模型對此文本不夠確定，結果僅供參考。

## 模型說明
- `roberta-base-openai-detector`，二分類標籤：
  - `Fake`: AI 生成
  - `Real`: 人類撰寫
- 輕量、CPU 友好，輸入長度截斷至 512 token 以保持速度。

## 已知限制
- 對極短文本或高度重寫內容，信心可能偏低。
- 非英文/中文的文本尚未特別優化。
- Hugging Face 線上下載模型需網路，若封閉環境請事先手動下載模型放入快取。

## 後續優化方向
- 替換更強的偵測模型（如近期針對 GPT-4/LLAMA 的偵測器）。
- 加入傳統特徵（困惑度、常見詞頻等）與 transformer 混合投票，提升魯棒性。
- 蒐集中英文真實/生成語料微調模型，以適配特定領域。
- 增加批次檔案處理與結果匯出（CSV）。
