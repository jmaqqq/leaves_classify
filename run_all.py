import sys
import csv
import subprocess
from pathlib import Path
from datetime import datetime

project_dir = Path(__file__).resolve().parent
train_script = project_dir / "train.py"
output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

SEED = 42

experiments = [
    {
        "name": "Baseline",
        "mode": "baseline",
        "scheduler": False
    },
    {
        "name": "Flip",
        "mode": "flip",
        "scheduler": False
    },
    {
        "name": "Aug",
        "mode": "aug",
        "scheduler": False
    },
    {
        "name": "Baseline-Cosine",
        "mode": "baseline",
        "scheduler": True
    },
    {
        "name": "Flip-Cosine",
        "mode": "flip",
        "scheduler": True
    },
    {
        "name": "Aug-Cosine",
        "mode": "aug",
        "scheduler": True
    }
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = output_dir / f"batch_summary_{timestamp}.csv"
results = []
for index, experiment in enumerate(experiments):

    name = experiment["name"]
    mode = experiment["mode"]
    use_scheduler = experiment["scheduler"]

    command = [
        sys.executable,
        str(train_script),
        "--mode",
        mode,
        "--seed",
        str(SEED)
    ]

    if use_scheduler:
        command.append("--scheduler")

    print("\n" + "=" * 60)
    print(f"开始实验 [{index + 1}/{len(experiments)}]")
    print(f"实验名称：{name}")
    print(f"实验模式：{mode}")
    print(f"余弦退火：{use_scheduler}")
    print(f"随机种子：{SEED}")
    print("=" * 60)

    start_time = datetime.now()

    result = subprocess.run(
        command,
        cwd=project_dir,
        check=False
    )

    end_time = datetime.now()

    if result.returncode == 0:
        status = "Success"
        print(f"\n{name} 实验完成！")
    else:
        status = "Failed"
        print(f"\n{name} 实验失败，退出代码：{result.returncode}")

    results.append({
        "name": name,
        "mode": mode,
        "scheduler": use_scheduler,
        "seed": SEED,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "return_code": result.returncode
    })

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

print("\n" + "=" * 60)
print("全部实验运行结束")
print("=" * 60)

for result in results:
    print(
        f"{result['name']:<20}"
        f"{result['status']}"
    )

print("\n实验总表保存位置：")
print(summary_path)
