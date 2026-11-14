import torch
from torch import nn
from pathlib import Path
from dataset import get_data_loaders
from resnet_18 import ResNet
import time
import pandas as pd
import argparse
from datetime import datetime


device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("当前设备：",device)

project_dir=Path(__file__).resolve().parent
data_dir=project_dir/"data"
parser=argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    choices=["baseline","aug"],
    default="baseline",
)
parser.add_argument(
    "--scheduler",
    action="store_true"
)
args=parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_path=project_dir/"outputs"
run_name = f"{timestamp}_{args.mode}"
if args.scheduler:
    run_name+="_cosine"
run_dir=output_path/ run_name
run_dir.mkdir(parents=True, exist_ok=False)
history_path = run_dir / "history.csv"
best_model_path = run_dir / "best_model.pth"


train_loader, val_loader, classes, class_to_idx = get_data_loaders(
    data_dir,
    batch_size=512,
    mode=args.mode
)

model=ResNet(num_class=len(classes)).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

def evaluate(model,val_loader,loss_fn,device):
    model.eval()
    # 在训练阶段，BN使用当前Batch的均值和方差进行归一化，同时更新运行统计量。
    # 在测试阶段，BN使用训练期间积累的运行均值和方差进行归一化，而不是被跳过。
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for images,labels in val_loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            loss=loss_fn(outputs,labels)
            total_loss += loss.item() * images.size(0)
            correct=(outputs.argmax(dim=1)==labels).sum().item()
            total_correct+=correct
            sample=images.size(0)
            total_samples+=sample
    val_loss=total_loss/total_samples
    val_acc=(total_correct/total_samples)*100
    return val_loss,val_acc

num_epochs=30
best_val_acc=.0
history=[]
scheduler = None
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_samples += images.size(0)
        total_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        total_correct += correct

    train_loss = total_loss / total_samples
    print("epoch loss:", train_loss)
    train_acc = total_correct / total_samples * 100
    print(f"Epoch[{epoch+1}/{num_epochs}]")
    print(f"Train Accuracy :{train_acc:.2f}%")
    print(f"Train loss:{train_loss:.4f}")
    val_loss, val_acc = evaluate(
        model, val_loader, loss_fn, device
    )
    epoch_time = time.time() - start_time

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_time": epoch_time
    })
    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")

    if val_acc>best_val_acc:
        best_val_acc=val_acc
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_to_idx": class_to_idx
        }, best_model_path)

    if scheduler is not None:
        scheduler.step()

        print(
            f"Next Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )


print("\n 训练完成")
print(f"最佳准确率：{best_val_acc:.2f}%")




