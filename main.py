import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from ortools.linear_solver import pywraplp  # Thêm OR-Tools

# Đường dẫn model
MODEL_PATH = "dog_cat_model.h5"

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Error", f"Cannot load model: {str(e)}")
    model = None


# Hàm tiền xử lý hình ảnh
def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Hàm tối ưu hóa kết quả phân loại
def optimize_classification(predictions):
    # Tạo solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise Exception("Solver not created.")

    # Biến quyết định
    x = [solver.BoolVar(f'x{i}') for i in range(len(predictions))]

    # Ràng buộc: Chỉ chọn tối đa một dự đoán (ví dụ)
    solver.Add(sum(x) == 1)

    # Hàm mục tiêu: Chọn dự đoán có xác suất cao nhất
    solver.Maximize(solver.Sum([predictions[i] * x[i] for i in range(len(predictions))]))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("Solver did not find an optimal solution.")

    # Trả về chỉ số của dự đoán được chọn
    for i in range(len(predictions)):
        if x[i].solution_value() > 0:
            return i
    return -1  # Nếu không chọn được


# Hàm mở và phân loại hình ảnh
def open_image(label_image, label_result):
    if model is None:
        messagebox.showerror("Error", "Model not loaded.")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            # Hiển thị hình ảnh
            img = Image.open(file_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            label_image.config(image=img_tk)
            label_image.image = img_tk

            # Tiền xử lý và dự đoán
            processed_img = preprocess_image(file_path)
            prediction = model.predict(processed_img)[0][0]

            # Tối ưu hóa kết quả dự đoán
            optimized_index = optimize_classification([1 - prediction, prediction])
            result = "Dog" if optimized_index == 1 else "Cat"

            # Hiển thị kết quả
            label_result.config(text=f"Result: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")


# Giao diện người dùng (GUI)
def main():
    # Tạo cửa sổ chính
    window = tk.Tk()
    window.title("Dog or Cat Classifier")
    window.geometry("450x500")
    window.configure(bg='#f4f4f9')  # Màu nền sáng

    # Tiêu đề
    title_label = tk.Label(window, text="Dog or Cat", font=("Segoe UI", 30, 'bold'), bg='#f4f4f9', fg='#3b3b3b')
    title_label.pack(pady=20)

    # Khung hiển thị hình ảnh và kết quả
    frame = tk.Frame(window, bg='#f4f4f9')
    frame.pack(pady=10)

    label_image = tk.Label(frame, bg='#f4f4f9')
    label_image.grid(row=0, column=0, padx=20, pady=10)

    label_result = tk.Label(frame, text="Result: ", font=("Segoe UI", 14), bg='#f4f4f9', fg='#3b3b3b')
    label_result.grid(row=1, column=0, pady=10)

    # Nút tải ảnh
    upload_button = tk.Button(window, text="Upload Image", command=lambda: open_image(label_image, label_result),
                               font=("Segoe UI", 14, 'bold'), bg="grey", fg="white", relief="flat", width=20, height=2)
    upload_button.pack(pady=30)

    # Chạy vòng lặp giao diện
    window.mainloop()


if __name__ == "__main__":
    main()
