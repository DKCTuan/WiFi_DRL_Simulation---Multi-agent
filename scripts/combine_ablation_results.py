"""
scripts/combine_ablation_results.py
=====================================
Gộp kết quả ablation đã chạy RỜI RẠC trên nhiều máy/tài khoản (mỗi máy chạy
1 phần --keys / --seeds của run_ablation_K10.py hoặc run_ablation_K10_hybrid.py)
thành file .txt tổng hợp chính thức (mean/std đủ số seed).

CÁCH DÙNG:
1. Trên mỗi máy, sau khi chạy xong phần việc được giao, lấy toàn bộ thư mục
   results/data/raw/ (chứa các file raw_K<K>_<tag>_<key>_seed<seed>.json) — đây là
   phần DUY NHẤT bạn cần copy về, không cần copy cả results/.
2. Gom hết các file raw_*.json từ mọi máy vào CHUNG một thư mục
   results/data/raw/ trên máy tổng hợp (đơn giản nhất: giải nén/copy đè, vì
   tên file luôn có đủ tag+key+seed nên không trùng nhau).
3. Chạy:
       python scripts/combine_ablation_results.py --k 10
   (mặc định gộp 2 tag "ablation_clean" và
   "ablation_clean_hybrid", đủ 3 seed
   [0,1,2] theo config.FINAL_SEEDS). Có thể chỉnh bằng --tags / --seeds nếu
   dùng tag hoặc seed khác mặc định.

Kết quả: ghi lại (đè) các file training_K<K>_<tag>_<key>.txt và
eval_summary_K<K>_<tag>_<key>.txt trong results/data/ bằng bản CHÍNH THỨC gộp
đủ seed, đồng thời in bảng so sánh baseline vs từng ablation cho mỗi tag.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._base import save_ablation_outputs
import config as _cfg
from training.utils import aggregate_results

DATA_DIR = "results/data"
RAW_PATTERN = re.compile(
    r"^raw_(?:K(?P<k>\d+)_)?(?P<tag>.+)_"
    r"(?P<key>baseline|no_ddqn|no_qmix|no_vib|with_vib|hybrid_vib|coop_cobf)_"
    r"seed(?P<seed>\d+)\.json$"
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--k", type=int, default=10,
                     help="Số STA mỗi AP cần tổng hợp (mặc định: 10)")
parser.add_argument("--tags", type=str, default="ablation_clean,ablation_clean_hybrid",
                     help="Danh sách tag cách nhau dấu phẩy (mặc định: cả Full-AI và Hybrid)")
parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in _cfg.FINAL_SEEDS),
                     help="Danh sách seed CẦN CÓ ĐỦ để coi là hoàn chỉnh (mặc định: config.FINAL_SEEDS)")
parser.add_argument("--raw-dir", type=str, default=os.path.join(DATA_DIR, "raw"),
                     help="Thư mục chứa các file raw_*.json đã gom về (mặc định: results/data/raw)")
args = parser.parse_args()

wanted_tags = set(t.strip() for t in args.tags.split(","))
wanted_seeds = set(int(s) for s in args.seeds.split(","))
K = args.k

if not os.path.isdir(args.raw_dir):
    print(f"❌ Không tìm thấy thư mục {args.raw_dir}. Hãy copy các file raw_*.json từ mọi máy vào đây trước.")
    sys.exit(1)

# Gom file theo (tag, key) -> {seed: result_dict}
grouped = {}
for path in glob.glob(os.path.join(args.raw_dir, "raw_*.json")):
    fname = os.path.basename(path)
    m = RAW_PATTERN.match(fname)
    if not m:
        print(f"  (bỏ qua file không đúng định dạng: {fname})")
        continue
    # Legacy raw files had no K in their name and were produced only by the
    # old K=10 combiner. New files always carry K to prevent K10/K12 overwrite.
    file_k = int(m.group("k")) if m.group("k") is not None else 10
    if file_k != K:
        continue
    tag, key, seed = m.group("tag"), m.group("key"), int(m.group("seed"))
    if tag not in wanted_tags:
        continue
    with open(path) as f:
        result = json.load(f)
    grouped.setdefault((tag, key), {})[seed] = result

if not grouped:
    print(f"❌ Không tìm thấy raw JSON nào khớp tag={sorted(wanted_tags)} trong {args.raw_dir}.")
    sys.exit(1)

LABELS = {
    "baseline": "Clean baseline (no VIB)",
    "no_ddqn":  "No Double DQN",
    "no_qmix":  "VDN (no learned mixer)",
    "no_vib":   "Legacy no VIB (beta=0)",
    "with_vib": "With VIB",
    "hybrid_vib": "Hybrid + WF + VIB (no cooperation)",
    "coop_cobf": "Wi-Fi 8 user-centric Co-BF + WF + VIB",
}

for tag in sorted(wanted_tags):
    keys_here = sorted({key for (t, key) in grouped if t == tag}, key=lambda k: list(LABELS).index(k) if k in LABELS else 99)
    if not keys_here:
        print(f"\n(không có dữ liệu nào cho tag='{tag}', bỏ qua)")
        continue

    print(f"\n{'='*70}\n  GỘP KẾT QUẢ — tag='{tag}' | K={K}\n{'='*70}")
    summary_rows = []
    for key in keys_here:
        seed_dict = grouped[(tag, key)]
        have_seeds = sorted(seed_dict.keys())
        missing = wanted_seeds - set(have_seeds)
        if missing:
            print(f"  ⚠️  [{key}] THIẾU seed {sorted(missing)} — chỉ gộp trên seed có sẵn {have_seeds}. "
                  f"Kết quả sẽ KHÔNG đầy đủ cho đến khi chạy nốt seed còn thiếu.")
        runs = [seed_dict[s] for s in have_seeds]
        agg = aggregate_results(runs)
        label = LABELS.get(key, key)
        thr, jfi, aps, ee = save_ablation_outputs(K, tag, key, label, have_seeds, agg, DATA_DIR)
        summary_rows.append((key, thr, jfi, aps, ee, have_seeds))

    print(f"\n  {'Ablation':<12}{'Throughput':>14}{'JFI':>10}{'Active-AP':>12}{'EnergyEff':>12}   Seeds")
    for key, thr, jfi, aps, ee, have_seeds in summary_rows:
        print(f"  {key:<12}{thr:>11.2f} Mbps{jfi:>10.3f}{aps:>12.2f}{ee:>12.3f}   {have_seeds}")

print(f"\n✓ Đã ghi file .txt chính thức vào {DATA_DIR}/ (đè lên bản 'xem nhanh' nếu có).")
