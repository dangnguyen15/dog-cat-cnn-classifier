import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ortools.linear_solver import pywraplp  # Thêm OR-Tools

def optimize_batch_selection(num_samples):
    # Tạo solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise Exception("Solver not created.")

    # Biến quyết định
    x = [solver.BoolVar(f'x{i}') for i in range(num_samples)]

    # Ràng buộc: Chọn một số lượng batch tối ưu (giả sử tối đa 100 mẫu)
    solver.Add(sum(x) <= 100)

    # Hàm mục tiêu: Tối ưu hóa (ví dụ) điểm trọng số giả định của từng mẫu
    weights = [1.0] * num_samples  # Điểm trọng số, có thể từ độ khó của mẫu
    solver.Maximize(solver.Sum([weights[i] * x[i] for i in range(num_samples)]))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("Solver did not find an optimal solution.")

    # Lấy danh sách mẫu được chọn
    selected_samples = [i for i in range(num_samples) if x[i].solution_value() > 0]
    return selected_samples

def create_and_train_model(train_dir, val_dir, model_path):
    # Tạo mô hình CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),  # Tránh overfitting
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Phân loại nhị phân
    ])

    # Compile mô hình
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Tăng cường dữ liệu
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
    )

    # Lấy số lượng mẫu và tối ưu hóa batch
    num_samples = train_generator.samples
    selected_samples = optimize_batch_selection(num_samples)
    print(f"Selected samples: {selected_samples}")

    # Huấn luyện mô hình
    model.fit(train_generator, epochs=10, validation_data=val_generator)

    # Lưu mô hình
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Huấn luyện và lưu mô hình
train_dir = "dataset/train"
val_dir = "dataset/val"
model_path = "dog_cat_model.h5"
create_and_train_model(train_dir, val_dir, model_path)
