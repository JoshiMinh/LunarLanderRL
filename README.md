# 🚀 LunarLanderRL: Reinforcement Learning (RL) Project

Dự án này tập trung vào việc triển khai và đánh giá hiệu suất của các thuật toán Học Tăng Cường (Reinforcement Learning) trên môi trường **LunarLander**. Đây là một đồ án nghiên cứu kết hợp giữa kỹ thuật phần mềm chuẩn hóa và khả năng trực quan hóa thực nghiệm.

---

## 🌎 Nguồn gốc & Bối cảnh kỹ thuật

### 1. Gymnasium là gì?
[Gymnasium](https://gymnasium.farama.org/) là một bộ thư viện tiêu chuẩn (Toolkit) mã nguồn mở dùng để phát triển và so sánh các thuật toán Reinforcement Learning. Nó cung cấp giao diện chuẩn hóa để "Agent" (Trí tuệ nhân tạo) có thể tương tác với "Environment" (Môi trường giả lập).

### 2. Tại sao là `LunarLander-v2`?
`LunarLander-v2` là một môi trường mô phỏng hạ cánh con tàu vũ trụ được phát triển bởi OpenAI. 
*   **Tính kế thừa**: Đây là phiên bản đã được tinh chỉnh (v2) để đảm bảo các quy luật vật lý và hệ thống tính điểm ổn định nhất.
*   **Hợp đồng kỹ thuật**: Trong lập trình, ID `LunarLander-v2` là một định danh cố định. Việc giữ đúng ID này là bắt buộc để thư viện Gymnasium có thể khởi tạo đúng các tham số môi trường chuẩn quốc tế.

---

## 🏗️ Kiến trúc Hybrid (Tối ưu cho Đồ án)

Dự án sử dụng mô hình **Hybrid (Lai)** để đáp ứng các tiêu chuẩn khắt khe của một đồ án đại học:

1.  **Core Engine (`.py` files)**: Chứa logic cốt lõi (Module hóa). Việc tách riêng code vào các file Python giúp mã nguồn sạch sẽ, dễ dàng Debug và chứng minh khả năng tổ chức dự án chuyên nghiệp.
2.  **Experimental Sandbox (`.ipynb` files)**: Sử dụng Jupyter Notebook để thực hiện các thí nghiệm, vẽ đồ thị so sánh (2D Graphs) và lưu lại kết quả huấn luyện. Đây là phần "báo cáo trực quan" quan trọng nhất để thuyết phục người xem.
3.  **Live Showcase (UI/Game)**: Tích hợp chế độ Render trực tiếp để trình diễn Agent chơi game ở tốc độ thực tế, tạo trải nghiệm như một trò chơi điện tử hoàn chỉnh.

---

## 📊 Thông số môi trường

| Thành phần | Đặc điểm chi tiết |
| :--- | :--- |
| **Môi trường** | `LunarLander-v2` (Định danh bắt buộc) |
| **Trạng thái (8)** | Tọa độ (X, Y), Vận tốc (X, Y), Góc, Vận tốc góc, Tiếp xúc chân. |
| **Hành động (4)** | Rời rạc: Tắt máy, Đẩy trái, Đẩy chính, Đẩy phải. |
| **Phần thưởng** | +100 cho hạ cánh, -100 cho va chạm, phạt nhẹ theo mức dùng nhiên liệu. |

---

## 🧠 Các thuật toán triển khai (Models)

Dự án so sánh 3 kiến trúc mạng nơ-ron sâu (Deep Q-Network):
*   **Vanilla DQN**: Nguyên bản mạng Q-learning sâu.
*   **Double DQN**: Khắc phục lỗi ước tính quá cao (Overestimation) bằng cách tách biệt mạng chọn và mạng đánh giá.
*   **Dueling DQN**: Chia mạng thành hai luồng độc lập: Giá trị trạng thái $V(s)$ và Lợi thế hành động $A(s, a)$.

---

## 📁 Cấu trúc thư mục đề xuất

```bash
LunarLanderRL/
├── core/               # Chứa logic tái sử dụng (Hybrid Core)
│   ├── model.py        # Các kiến trúc mạng nơ-ron (PyTorch)
│   ├── agent.py        # Logic DQN, Double DQN, Dueling DQN
│   └── memory.py       # Bộ nhớ đệm (Experience Replay)
├── experiments/        # Thí nghiệm & Đồ thị
│   └── comparison.ipynb # So sánh 3 thuật toán & Vẽ đồ thị 2D
├── demo.py             # Script trình diễn UI (Live Render)
├── models/             # Lưu trữ trọng số (.pth) sau khi học xong
└── README.md           # Hướng dẫn chi tiết
```

---

## 🚀 Quy trình triển khai (Step-by-Step)

### Bước 1: Chuẩn bị môi trường
Cài đặt toolkit và framework:
```bash
pip install gymnasium[box2d] torch matplotlib tqdm
```

### Bước 2: Thiết lập Logic lõi (Core)
Xây dựng các lớp `DQN`, `DoubleDQN` trong thư mục `core/`. Luôn kiểm tra thiết bị tính toán:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Bước 3: Huấn luyện & Phân tích (Notebook)
Mở `comparison.ipynb` để chạy huấn luyện. Tại đây, bạn sẽ sử dụng `matplotlib` để vẽ đồ thị Reward và Loss qua từng Episode. Đây là cơ sở để đánh giá thuật toán nào hiệu quả hơn.

### Bước 4: Trình diễn (Live Demo)
Sử dụng `demo.py` để quan sát Agent xuất sắc nhất trình diễn kỹ năng hạ cánh:
```python
# Chạy demo với chế độ "human" để mở cửa sổ game
env = gym.make('LunarLander-v2', render_mode='human')
```

---

## 📝 Giấy phép
Dự án được thực hiện phục vụ mục đích nghiên cứu học thuật và đồ án đại học.
