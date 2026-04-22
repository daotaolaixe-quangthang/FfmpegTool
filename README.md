# 🎬 FfmpegTool — Hướng Dẫn Sử Dụng

> **Video Frame Extractor & Smart Filter**  
> Trích xuất frame từ video TikTok/YouTube → Lọc blur → Lọc trùng → Chọn frame đẹp nhất

---

## 📁 Cấu Trúc Thư Mục

```
E:\FfmpegTool\
├── start.bat           ← Double-click để mở Web UI
├── app.py              ← Flask web server
├── main.py             ← CLI entry point
├── extractor.py        ← Trích xuất frame (FFmpeg)
├── filters.py          ← Lọc blur + lọc trùng (pHash / SSIM)
├── scorer.py           ← Chấm điểm thẩm mỹ frame
├── downloader.py       ← Tải video từ TikTok/YouTube (yt-dlp)
├── reporter.py         ← Xuất JSON report + HTML gallery
├── config.json         ← Cấu hình mặc định (edit thoải mái)
├── requirements.txt    ← Danh sách thư viện Python
└── templates/
    └── index.html      ← Giao diện Web UI
```

---

## ⚡ Cài Đặt Nhanh

### 1. Yêu cầu hệ thống
- **Python** 3.10+ — [python.org](https://python.org)
- **FFmpeg** — cài bằng winget:
  ```bat
  winget install Gyan.FFmpeg
  ```
  Sau khi cài, mở PowerShell mới để PATH được cập nhật.

### 2. Cài thư viện Python
```bat
pip install -r E:\FfmpegTool\requirements.txt
```

### 3. Kiểm tra FFmpeg
```bat
ffmpeg -version
```
Kết quả trả về version là OK.

---

## 🖥️ Cách 1 — Giao Diện Web UI (Khuyến nghị)

### Khởi động

**Double-click vào file:**
```
E:\FfmpegTool\start.bat
```

Browser tự mở `http://localhost:5000`

---

### Hướng dẫn từng phần trên UI

#### 📌 INPUT SOURCE — Chọn nguồn video

| Tab | Dùng khi |
|---|---|
| **Local File** | Đã có video trên máy (mp4, mkv, mov...) |
| **URL (TikTok/YT)** | Muốn tải thẳng từ TikTok / YouTube / Instagram |
| **Batch Folder** | Xử lý cả thư mục có nhiều video cùng lúc |

- **Video File Path** — Điền đường dẫn đầy đủ, ví dụ: `G:\Videos\clip.mp4`
- **Output Folder** — Thư mục lưu kết quả, ví dụ: `G:\Frames`

> 💡 Kết quả mỗi video sẽ được lưu vào `Output\<tên_video>\`

---

#### ⚙️ EXTRACTION — Cách trích xuất frame

| Cài đặt | Mô tả |
|---|---|
| **Mode: FPS** | Lấy N frame mỗi giây. Mặc định 5fps → 1 frame/0.2 giây |
| **Mode: Scene** | Phát hiện cảnh cắt → lấy 1 frame giữa mỗi cảnh (thông minh hơn) |
| **Frames Per Second** | Slider 1–30. Thấp = ít frame hơn, nhanh hơn. Cao = nhiều frame, chậm hơn |

> **Khuyến nghị:** Video TikTok ngắn dùng `fps=5`. Video dài nhiều cảnh cắt dùng `Scene`.

---

#### 🔍 FILTERS — Lọc frame

| Cài đặt | Mô tả | Khuyến nghị |
|---|---|---|
| **Blur Threshold** | Ngưỡng Laplacian variance. Thấp hơn = xóa nhiều frame nhòe hơn | 60–100 |
| **Max Similarity** | Frame có similarity ≥ mức này bị xóa là trùng lặp | 0.65–0.75 |
| **Dedup Method: pHash** | Nhanh, phù hợp hầu hết trường hợp | **Mặc định** |
| **Dedup Method: SSIM** | Chính xác hơn, chậm hơn ~5–10x. Dùng khi <500 frame | Video chất lượng cao |

**Hiểu ngưỡng Similarity:**
```
0.70 = giữ frame nếu khác nhau > 30% so với frame đã giữ  ← Mặc định
0.60 = giữ frame nếu khác nhau > 40% (chặt hơn, ít frame hơn)
0.85 = giữ frame nếu khác nhau > 15% (lỏng hơn, nhiều frame hơn)
```

---

#### ⭐ AESTHETIC SCORER — Chọn frame đẹp nhất

| Cài đặt | Mô tả |
|---|---|
| **Enable Scorer** | Bật/tắt tính năng chấm điểm |
| **Top N Frames** | Số frame tốt nhất cần chọn (copy vào thư mục `top_frames/`) |

**Scorer chấm điểm theo 5 tiêu chí:**

| Tiêu chí | Trọng số | Đo lường gì |
|---|---|---|
| Sharpness | 35% | Độ sắc nét (Laplacian variance) |
| Colorfulness | 20% | Màu sắc phong phú, rực rỡ |
| Brightness | 20% | Độ sáng vừa phải (không quá tối/sáng) |
| Contrast | 15% | Độ tương phản |
| Composition | 10% | Bố cục (edge trọng tâm theo rule of thirds) |

---

#### 🔧 OPTIONS

| Tùy chọn | Mô tả |
|---|---|
| **Keep Raw Frames** | Giữ lại frame thô trước khi lọc (mặc định tự xóa để tiết kiệm ổ cứng) |
| **Skip HTML Preview** | Không tạo file preview.html (nhanh hơn khi xử lý batch lớn) |

---

#### 📊 Sau khi chạy xong

Kết quả hiện ngay dưới terminal:

| Card | Ý nghĩa |
|---|---|
| **Raw Frames** | Tổng frame thô FFmpeg extract ra |
| **Blurry Removed** | Số frame bị xóa do nhòe |
| **Duplicates Removed** | Số frame bị xóa do trùng |
| **Unique Kept** | Số frame sạch, dùng được |

**Nút hành động:**
- `📁 Open Output Folder` — Mở thư mục kết quả trong Explorer
- `🖼 View HTML Preview` — Mở gallery hình ảnh trong browser
- `↺ Run Again` — Chạy lại với cài đặt khác

---

#### 📂 Cấu trúc thư mục output

```
G:\Frames\
└── #disneyworldvlogs_48\
    ├── unique_frames\          ← Frame sạch (không blur, không trùng)
    │   ├── unique_0000.jpg
    │   ├── unique_0001.jpg
    │   └── ...
    ├── top_frames\             ← Frame đẹp nhất (nếu bật Scorer)
    │   ├── top_001_score0.88.jpg
    │   ├── top_002_score0.87.jpg
    │   └── ...
    ├── preview.html            ← Gallery visual từng video (mở bằng browser)
    ├── report.json             ← Báo cáo chi tiết
    └── score_report.json       ← Điểm thẩm mỹ từng frame (nếu bật Scorer)
_batch_preview.html             ← ✨ Master gallery tổng hợp TẤT CẢ video (root)
```

---

## 💻 Cách 2 — Command Line (CLI)

### Cú pháp cơ bản

```bat
python E:\FfmpegTool\main.py --input <video> --output <thư_mục>
```

### Tất cả options

```
Nhóm INPUT (chọn 1 trong 3):
  --input  / -i   <path>    Đường dẫn video local
  --url    / -u   <url>     URL TikTok / YouTube / Instagram
  --batch  / -b   <folder>  Thư mục chứa nhiều video

Nhóm OUTPUT:
  --output / -o   <folder>  Thư mục lưu kết quả (BẮT BUỘC)

Nhóm EXTRACTION:
  --mode          fps|scene  Chế độ trích xuất (mặc định: fps)
  --fps           <số>       Số frame/giây, mode fps (mặc định: 5)

Nhóm FILTER:
  --blur          <số>       Ngưỡng blur Laplacian (mặc định: 80)
  --sim           <0.0-1.0>  Ngưỡng similarity tối đa (mặc định: 0.70)
  --method        phash|ssim Phương pháp dedup (mặc định: phash)

Nhóm SCORER:
  --top           <số>       Bật scorer, chọn top-N frame đẹp nhất

Nhóm OUTPUT OPTIONS:
  --keep-raw                 Giữ frame thô sau khi lọc
  --no-html                  Không tạo HTML preview
  --no-report                Không tạo JSON report
  --gen-batch-html           Tạo (lại) file _batch_preview.html từ output folder có sẵn
  --clean-empty              Xóa các sub-folder không có unique_frames/ (dọn folder rỗng)
```

---

### Ví Dụ Thực Tế

#### Video local, cài đặt mặc định
```bat
python main.py --input "G:\Videos\clip.mp4" --output "G:\Frames"
```

#### Tải từ TikTok rồi xử lý
```bat
python main.py --url "https://www.tiktok.com/@user/video/..." --output "G:\Frames"
```

#### Dùng Scene mode (thông minh, ít frame hơn nhưng đúng hơn)
```bat
python main.py --input "clip.mp4" --output "G:\Frames" --mode scene
```

#### Filter chặt hơn (xóa nhiều frame nhòe + frame trùng hơn)
```bat
python main.py --input "clip.mp4" --output "G:\Frames" --blur 120 --sim 0.60
```

#### Dùng SSIM (chính xác hơn, chậm hơn)
```bat
python main.py --input "clip.mp4" --output "G:\Frames" --method ssim
```

#### Bật Scorer — chọn top 25 frame đẹp nhất
```bat
python main.py --input "clip.mp4" --output "G:\Frames" --top 25
```

#### Full combo — tất cả options
```bat
python main.py ^
  --input "G:\Downloads\Video\clip.mp4" ^
  --output "G:\Frames" ^
  --mode scene ^
  --blur 100 ^
  --sim 0.65 ^
  --method phash ^
  --top 30 ^
  --keep-raw
```

#### Batch — xử lý cả thư mục video
```bat
python main.py --batch "G:\Downloads\Videos\" --output "G:\Frames"
```

---

### Bảng Chọn Cài Đặt Theo Use Case

| Tình huống | fps | blur | sim | method | top |
|---|---|---|---|---|---|
| TikTok ngắn (15–60s), chất lượng cao | 5 | 80 | 0.70 | phash | 20 |
| YouTube dài (5–30 phút) | 2–3 | 80 | 0.65 | phash | 30–50 |
| Video nhiều cảnh cắt nhanh | scene | 80 | 0.70 | phash | 20 |
| Muốn frame thật sự độc đáo (ít) | 3 | 100 | 0.55 | ssim | 15 |
| Muốn nhiều frame để chọn tay | 10 | 60 | 0.80 | phash | — |

---

## ⚙️ Tùy Chỉnh Config Mặc Định

Thay vì gõ options mỗi lần, sửa file `config.json`:

```json
{
  "extraction": {
    "fps": 5,
    "mode": "fps",
    "scene_threshold": 27.0,
    "jpeg_quality": 2
  },
  "filter": {
    "blur_threshold": 80.0,
    "similarity_threshold": 0.70,
    "dedup_method": "phash",
    "phash_size": 16
  },
  "scorer": {
    "enabled": false,
    "top_n": 30,
    "save_score_report": true
  },
  "output": {
    "keep_raw": false,
    "generate_html_preview": true,
    "preview_columns": 5,
    "report_json": true
  }
}
```

> **Lưu ý:** CLI arguments luôn **ghi đè** config.json.

---

## 🔄 Quy Trình Hoàn Chỉnh (Workflow)

```
[Bước 1] Tải video TikTok/YouTube về G:\Downloads\Video\
          ↓
[Bước 2] Chạy FfmpegTool
          • Mode: FPS 5 | Blur: 80 | Sim: 0.70 | Top: 25
          ↓
[Bước 3] Mở thư mục top_frames\ (25 frame đẹp nhất)
          ↓
[Bước 4] Gửi vào AI Agent (Claude / Gemini / ChatGPT)
          • Attach 25 frame ảnh + Video Brief
          • AI tạo Production Sheet (SCREEN 01...N)
          ↓
[Bước 5] Từng PROMPT → Veo3 / Grok Video / Kling → clip_XX.mp4
          ↓
[Bước 6] VOICE SCRIPT → ElevenLabs → narration.mp3
          ↓
[Bước 7] FFmpeg assembly → final_video.mp4
```

---

## ✨ Tính Năng Mới

### 📄 Master Batch HTML Preview

Sau mỗi lần chạy batch (nhiều video), tool tự động tạo file **`_batch_preview.html`** tại root của output folder.

| Tính năng | Mô tả |
|---|---|
| **Tổng hợp 1 trang** | Hiển thị tất cả frame của tất cả video trong 1 file HTML duy nhất |
| **Section thu gọn** | Mỗi video là 1 section — click header để collapse/expand |
| **Lightbox** | Click bất kỳ ảnh nào → mở full size, nhấn `Esc` để đóng |
| **Relative path** | Không embed base64 → file nhỏ, tải nhanh dù có hàng trăm frame |
| **Tự động** | Không cần làm gì thêm — tạo tự động sau khi batch xong |

> 💡 Mở `_batch_preview.html` để bao quát toàn bộ frame của tất cả video cùng lúc thay vì phải vào từng sub-folder.

---

### 🛠️ 2 CLI Utility Mới

#### `--clean-empty` — Dọn folder rỗng

Xóa các sub-folder trong output không có `unique_frames/` (thường xảy ra khi video không extract được frame nào).

```bat
python main.py --input dummy --output "G:\...\story1-frame" --clean-empty
```

Ví dụ output:
```
[CLEAN] Removed empty folder: 9645574-hd_1080_1920_25fps
[CLEAN] Done — removed 1 empty folder(s).
```

#### `--gen-batch-html` — Tạo lại Master HTML

Tạo (hoặc tạo lại) file `_batch_preview.html` từ output folder có sẵn mà **không cần chạy lại pipeline**. Hữu ích khi bạn đã xử lý xong và muốn regenerate gallery.

```bat
python main.py --input dummy --output "G:\...\story1-frame" --gen-batch-html
```

Ví dụ output:
```
[REPORT] Batch master HTML: G:\Pictures\Camera Roll\story1-frame\_batch_preview.html
[OK] Master batch HTML generated: G:\...\story1-frame\_batch_preview.html
```

> **Lưu ý:** Cả hai flags `--clean-empty` và `--gen-batch-html` là **standalone** — chạy xong tự thoát, không cần truyền input video thật cho `--input`.

---

## 🐛 Khắc Phục Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách fix |
|---|---|---|
| `ffmpeg: command not found` | FFmpeg chưa trong PATH | Mở PowerShell mới sau khi cài FFmpeg |
| `ModuleNotFoundError: cv2` | Chưa cài requirements | `pip install -r requirements.txt` |
| `[ERROR] No frames extracted` | Path video sai hoặc video bị lỗi | Kiểm tra lại path, thử mở video bằng player |
| `yt-dlp: HTTP Error 403` | TikTok/YT chặn download | Cập nhật yt-dlp: `pip install -U yt-dlp` |
| Browser không tự mở | Firewall chặn | Mở tay: `http://localhost:5000` |
| Preview HTML load chậm | Quá nhiều frame (>150) | HTML tự giới hạn 150 frame đầu |

---

## 📝 Ghi Chú Kỹ Thuật

- **pHash vs SSIM**: pHash nhanh hơn ~5–10x, đủ tốt cho 99% trường hợp. SSIM dùng khi cần độ chính xác cao hơn (phim chất lượng 4K, frame rất giống nhau về nội dung nhưng khác ánh sáng).
- **Blur threshold**: Giá trị Laplacian variance. Video 1080p thường cần threshold cao hơn (100–150) so với 720p (60–80).
- **HTML preview**: Giới hạn 150 frame đầu để tránh file quá lớn. Tất cả frame vẫn có trong thư mục `unique_frames\`.
- **Scorer**: Không phải "frame đẹp nhất về nghệ thuật" — mà là frame có chất lượng kỹ thuật tốt nhất (sắc nét, màu tốt, không quá tối/sáng). Bạn vẫn cần xem lại và chọn tay.
