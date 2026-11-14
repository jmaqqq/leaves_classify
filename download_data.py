import kagglehub
from pathlib import Path

kagglehub.login()
project_dir=Path(__file__).resolve().parent
data_dir=project_dir / "data"
path = kagglehub.competition_download(
    "classify-leaves",
    output_dir=str(data_dir)
)

print("Path to competition files:", path)