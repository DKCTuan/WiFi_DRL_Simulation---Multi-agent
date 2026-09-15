# eval_K4_K6.py — đã deprecate

## Lý do

Script này được thêm vào lúc kết quả K=4/K=6 trông bất thường, với mục đích
eval lại kỹ hơn (100 episodes thay vì 30). Nhưng nó vô tình tạo ra **sự
không nhất quán về phương pháp đo** so với K=8, 10, 12, 14:

| | K=4, K=6 (script này) | K=8, 10, 12, 14 (`_base.py::run_k`) |
|---|---|---|
| Checkpoint dùng | "best" (chọn theo `eval_score` cao nhất lúc train) | trung bình 5 checkpoint **cuối** quá trình train |
| Số episode/seed | 100 | 30 |
| Cách tính std | `np.std()` thô trên 3 giá trị (3 seed) | trung bình của 5 giá trị std (mỗi giá trị là std qua 3 seed tại 1 checkpoint) → std đã làm mượt |

→ Hai cách đo này không thể so sánh ngang hàng trên cùng một biểu đồ
`eval_*_vs_K`. Đây là nguyên nhân chính khiến error bar ở K=4 phình to bất
thường trong `fig3`/`fig4`.

## Đã sửa thế nào

Xóa bước eval lại này khỏi pipeline. `run_v3_K4.py` và `run_v3_K6.py` (dùng
chung `_base.py::run_k`) đã tự sinh ra đúng `eval_summary_K4_*.txt` và
`eval_summary_K6_*.txt` theo cùng phương pháp với K=8..14 ngay trong lúc
train — không cần bước eval lại riêng nữa.

## Cần làm gì tiếp theo

1. Nếu bạn đã từng chạy `eval_K4_K6.py` và nó **ghi đè** lên
   `results/data/eval_summary_K4_*.txt` / `eval_summary_K6_*.txt`, các file
   đó hiện đang chứa số liệu theo phương pháp cũ (sai lệch).
   → Cần train lại K=4, K=6 bằng `scripts/run_v3_K4.py` và
   `scripts/run_v3_K6.py` để sinh lại đúng `eval_summary` (phương pháp
   "trung bình 5 checkpoint cuối"), rồi mới chạy `plot_final.py`.
2. Nếu bạn còn giữ bản gốc `eval_summary_K4_*.txt` / `eval_summary_K6_*.txt`
   từ trước khi chạy `eval_K4_K6.py` (vd. trong bản backup/zip cũ), chỉ cần
   khôi phục lại 2 file đó vào `results/data/` và chạy thẳng
   `plot_final.py` — không cần train lại.

Nếu muốn dùng 100 episodes/eval cho toàn bộ 6 giá trị K để giảm nhiễu
(thay vì 30), cách đúng là sửa `config.EVAL_EPISODES` (áp dụng đồng đều cho
mọi K), chứ không nên patch riêng cho từng K.
