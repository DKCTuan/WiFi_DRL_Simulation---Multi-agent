# WiFi DRL Simulation — core

Nhánh core gồm hai chế độ: **AI-only** chọn CCA và công suất; **AI+WF** chọn
CCA rồi water-filling phân bổ công suất. Đây là bản nền tảng để phát triển
và kiểm thử, chưa phải benchmark cuối của nghiên cứu dịch vụ cố định.

## Cài và chạy

Yêu cầu Python 3.12. Tạo virtual environment và cài dependencies:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

Sau khi tạo môi trường, kích hoạt `.venv` trước lệnh cài/chạy: trên Windows
dùng `.venv\Scripts\Activate.ps1`; trên Linux/macOS dùng `source .venv/bin/activate`.
`python main.py` chỉ kiểm runtime của cả hai chế độ trong 12 slot bằng mạng
chưa học, không đánh giá chất lượng thuật toán và không khởi động train dài.

Huấn luyện khi cần, dùng cấu hình chung trong `config.py`:

```sh
python main.py --mode train --arm ai_only --episodes 25 --seed 0
python main.py --mode train --arm ai_wf --episodes 25 --seed 0
```

Output nằm trong `results/`, được Git bỏ qua. Resume dùng lại cùng experiment
name và cấu hình/source, ví dụ thêm `--resume --episodes 50`. Checkpoint kiểm
signature; không nạp checkpoint nghiên cứu cũ sang source core này.

## Phần được giữ

- `env/`: topology, channel/path-loss, mobility tùy chọn, seeded slotted
  contention, SINR/Shannon, ordinary bounded WF, rate/JFI và observation.
- `agent/`: DDQN, encoder và QMIX mixer, shared replay.
- `training/`: train/evaluate, seeding, scalar objective và checkpoint atomic.
- `utils/`: log và plotting dùng chung.
- `tests/`: kiểm PHY/công suất, contention, train/evaluation và resume.

CCA là action của agent và có thể đổi theo slot. Mỗi slot thực hiện admission
một lần trước khi áp dụng phân bổ công suất. Ngân sách tổng 0,18 W và peak
0,10 W/AP dùng chung cho cả hai chế độ.

## Phạm vi mô hình và cấu hình

Bản core giữ các mặc định generic hiện có: demand cap tắt, reward legacy,
chia đều airtime trong BSS, mobility và multi-layout training bật. Cấu hình
static 6 AP × 10 STA, demand 433,5 Mbps, scheduler tái sử dụng airtime và score
trên demand-normalized temporal JFI của pilot nghiên cứu **không được tự động
gộp vào lần dọn core này**. Không dùng output mặc định để diễn giải kết quả pilot.

`env/user_demand.py` là tiện ích nhu cầu tùy chọn. Khi bật demand, core cắt
rate theo nhu cầu; chưa tái sử dụng phần airtime dư. `jfi_time_average` trong
core tính trên throughput tích lũy, không phải satisfaction chia cho demand.
Những khác biệt này được ghi rõ để tránh gộp các protocol không tương đương.

Slotted sensing/SINR là abstraction, chưa có queue, delay, retry hay đầy đủ
IEEE 802.11 MAC. WF dùng channel costs cố định tại snapshot; không khẳng định
nó tối ưu score dịch vụ 50/50.

## Nhánh cooperative

Cooperative đang phát triển riêng. Nhánh core không có bộ chọn cặp, trao đổi
thông tin, guard dịch vụ, constrained WF, invitation hoặc mô hình Co-SR packet.
Tham số tương thích `cooperation_enabled=False` và một số trường log bằng 0
được giữ để giảm xáo trộn schema; truyền `True` bị từ chối ngay.
Notebook, kết quả, ZIP, checkpoint và cache không được đưa vào release core.

Xem [phạm vi cleanup](docs/CORE_CLEANUP.md) và
[manifest nguồn](docs/CORE_SOURCE_MANIFEST.json).
