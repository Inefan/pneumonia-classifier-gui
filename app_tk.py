import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import os
import sys

def resource_path(relative_path):
    """Отримує шлях до ресурсу, працює і в .exe"""
    try:
        base_path = sys._MEIPASS 
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_small(pretrained=True)
model.classifier[3] = torch.nn.Linear(1024, 2)
model.load_state_dict(torch.load(resource_path("BEST_MDL.pth3"), map_location=device))
model.eval().to(device)
classes = ["NORMAL", "PNEUMONIA"]

tta_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((224, 224))
    tk_img = ImageTk.PhotoImage(img_resized)
    image_label.config(image=tk_img)
    image_label.image = tk_img
    tta_runs = 5
    outputs_sum = 0


    with torch.no_grad():
        for _ in range(tta_runs):
            tta_img = tta_transforms(img)
            tta_img = tta_img.unsqueeze(0).to(device)
            outputs = model(tta_img)
            outputs_sum += torch.softmax(outputs, dim=1)


    avg_output = outputs_sum / tta_runs
    predicted_idx = avg_output.argmax(1).item()
    confidence = avg_output[0][predicted_idx].item()
    result_label.config(text=f"Результат: {classes[predicted_idx]} ({confidence*100:.2f}%)")

def show_code_window():
    top = tk.Toplevel(root)
    top.title("Код моделі")
    top.geometry("600x500")
    text_widget = tk.Text(top, wrap="word", font=("Courier", 10))
    text_widget.pack(expand=True, fill="both")
    try:
        with open(resource_path("model_code.txt"), "r", encoding="utf-8") as f:
            text_widget.insert("1.0", f.read())
    except Exception as e:
        text_widget.insert("1.0", f"Помилка: {e}")


root = tk.Tk()
root.title("Класифікатор пневмонії")
root.geometry("300x400")

btn = tk.Button(root, text="Вибрати зображення", command=classify_image)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Результат: ", font=("Arial", 14))
result_label.pack(pady=10)

btn_code = tk.Button(root, text="Показати код моделі", command=show_code_window)
btn_code.pack(pady=5)
root.mainloop()