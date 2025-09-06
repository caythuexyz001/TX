# TaiXiu Analyzer (FastAPI + Markov + Patterns)

> Phân tích cầu Tài/Xỉu (Markov bậc 1, run/alt/block patterns) + Dashboard tối giản. **Chỉ phục vụ mục đích thống kê/tham khảo.**

## Chạy local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
cp .env.example .env
uvicorn app.api.main:app --reload