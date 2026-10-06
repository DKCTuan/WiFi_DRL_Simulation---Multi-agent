# Core cleanup — 06/10/2026

Bản core được tách trong worktree Git riêng từ commit gốc 48672c1 và các
module dùng chung của checkout nghiên cứu. Checkout nghiên cứu, cooperative,
adapter, raw results và checkpoint tại máy gốc không bị sửa hoặc xóa.

Giữ DDQN/QMIX, shared replay, CCA contention, channel, ordinary WF, generic
train/evaluate và checkpoint. Tách engine cooperative và các entry point
thí nghiệm; loại bytecode đã bị Git track. Thêm requirements, CLI smoke/train,
kiểm thử core và CI. CLI mặc định chạy smoke ngắn, tránh vô tình train 1200 episode.

Không đổi reward/PHY/ordinary power allocation để thay xếp hạng hai nhánh.
Không đưa scheduler thử nghiệm hay invitation vào mặc định. Protocol/source
signature mới giúp từ chối resume checkpoint của nghiên cứu sang release này.
Một số trường log coordination bằng 0 là schema tương thích, không phải engine.

`CORE_SOURCE_MANIFEST.json` ghi hash input trước khi tách; các file ở checkout
nghiên cứu được kiểm không đổi sau khi tạo bản core. Các test scalar/vectorized
PHY, admission một lần, power bounds và resume kiểm bản core đã tách.

## Xác minh trước push

25/25 kiểm thử đạt. Cả hai arm thực hiện optimizer update; khôi phục từ cuối
episode cho ra weights/replay/throughput khớp với chạy liên tục. CLI mặc định
chạy 12 slot/arm và qua kiểm runtime/công suất.

So riêng 800 slot với cùng action, seed, topology và demand on/off giữa bản
core đã tách và code nghiên cứu có cooperative tắt: observation, reward,
rate/JFI, admission và công suất khớp chính xác. Phép thử này kiểm việc tách
module không làm đổi hai baseline ở các trace này; không phải bằng chứng hội
tụ hay ranking. Cấu hình runtime và hash trace nằm trong CORE_VERIFICATION.json.
