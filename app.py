from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from werkzeug.utils import secure_filename
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import functools
from datetime import datetime
from PIL import Image

app = Flask(__name__)
app.secret_key = 'ini_rahasia_yang_aman_123'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.jinja_env.add_extension('jinja2.ext.do')

# Fungsi koneksi database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def ensure_columns_exist():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Cek apakah kolom sudah ada di sample_foto
    cursor.execute("PRAGMA table_info(sample_foto)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    if "jenis_kerusakan" not in existing_columns:
        conn.execute("ALTER TABLE sample_foto ADD COLUMN jenis_kerusakan TEXT")

    # Cek kolom di hasil_analisa
    cursor.execute("PRAGMA table_info(hasil_analisa)")
    hasil_columns = [col[1] for col in cursor.fetchall()]

    if "tingkat_kerusakan" not in hasil_columns:
        conn.execute("ALTER TABLE hasil_analisa ADD COLUMN tingkat_kerusakan TEXT")

    if "luas_total" not in hasil_columns:
        conn.execute("ALTER TABLE hasil_analisa ADD COLUMN luas_total REAL")

    if "total_deduct" not in hasil_columns:
        conn.execute("ALTER TABLE hasil_analisa ADD COLUMN total_deduct REAL")

    # === Cek & Tambah Kolom di hasil_detail_kerusakan ===
    cursor.execute("PRAGMA table_info(hasil_detail_kerusakan)")
    detail_columns = [col[1] for col in cursor.fetchall()]

    for kolom in ["x1", "y1", "x2", "y2"]:
        if kolom not in detail_columns:
            conn.execute(f"ALTER TABLE hasil_detail_kerusakan ADD COLUMN {kolom} REAL")

    conn.commit()
    conn.close()

# Inisialisasi database jika belum ada
def init_db():
    conn = get_db_connection()

    # Tabel untuk data training
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sample_foto (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            labeled INTEGER DEFAULT 0
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS kerusakan_detail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            jenis TEXT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            FOREIGN KEY (sample_id) REFERENCES sample_foto(id)
        )
    ''')

    # Tabel untuk data hasil analisa
    conn.execute('''
        CREATE TABLE IF NOT EXISTS folder (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_folder TEXT NOT NULL,
            informasi TEXT,
            tanggal_dibuat TEXT NOT NULL
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS hasil_analisa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            nilai_pci REAL,
            tanggal_analisa TEXT,
            FOREIGN KEY (folder_id) REFERENCES folder(id)
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS hasil_detail_kerusakan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hasil_id INTEGER NOT NULL,
            jenis_kerusakan TEXT,
            tingkat_kerusakan TEXT,
            luas_persen REAL,
            deduct_value REAL,
            FOREIGN KEY (hasil_id) REFERENCES hasil_analisa(id)
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')) DEFAULT 'user'
        )
    ''')

    # Cek jika ada user admin
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    admin_exists = cursor.fetchone()

    if not admin_exists:
        # Tambah admin default
        hashed_password = generate_password_hash('admin123', method='pbkdf2:sha256')
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     ('admin', hashed_password, 'admin'))

    conn.commit()
    conn.close()

    ensure_columns_exist()

init_db()

# Decorator untuk memeriksa apakah pengguna sudah login
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

# Decorator untuk memeriksa apakah pengguna adalah admin
def admin_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if session.get('role') != 'admin':
            flash("You do not have permission to access this page.", "danger")
            return redirect(url_for('index'))
        return view(**kwargs)
    return wrapped_view

# Membuat data session tersedia di semua template
@app.context_processor
def inject_session():
    return dict(session=session)

@app.route("/")
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for("daftar_folder"))

# Halaman labeling
from collections import Counter

@app.route('/labeling')
@login_required
def labeling():
    conn = get_db_connection()
    sample_rows = conn.execute("SELECT * FROM sample_foto ORDER BY id DESC").fetchall()
    conn.close()

    result_images = []
    jenis_counter = Counter()

    for sample in sample_rows:
        filename = sample["filename"]
        image_path = os.path.join("static", "uploads", filename)

        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except:
            continue

        conn2 = get_db_connection()
        box_rows = conn2.execute(
            "SELECT x1, y1, x2, y2, jenis FROM kerusakan_detail WHERE sample_id = ?",
            (sample["id"],)
        ).fetchall()
        conn2.close()

        boxes = []
        for b in box_rows:
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"], b["jenis"]])
            jenis_counter[b["jenis"]] += 1

        result_images.append({**dict(sample), "boxes": boxes})

    total_sample = len(result_images)

    return render_template(
        "labeling.html",
        images=result_images,
        total_sample=total_sample,
        jenis_counter=jenis_counter
    )

# Upload gambar sample untuk labeling
@app.route("/upload_sample", methods=["POST"])
@login_required
def upload_sample():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    image = request.files.get("image")
    
    if image and allowed_file(image.filename):
        import uuid
        ext = image.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        conn = get_db_connection()
        conn.execute("INSERT INTO sample_foto (filename, uploaded_at, labeled) VALUES (?, ?, ?)",
                     (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0))
        conn.commit()
        conn.close()
        return redirect(url_for("labeling"))
    else:
        # Redirect ke halaman labeling dengan query ?upload=error
        return redirect(url_for("labeling", upload="error"))

@app.route("/folders")
@login_required
def daftar_folder():
    conn = get_db_connection()
    folders = conn.execute('''
        SELECT 
            folder.id,
            folder.nama_folder,
            folder.informasi,
            folder.tanggal_dibuat,
            COUNT(hasil_analisa.id) AS jumlah_foto
        FROM folder
        LEFT JOIN hasil_analisa ON folder.id = hasil_analisa.folder_id
        GROUP BY folder.id
        ORDER BY folder.tanggal_dibuat DESC
    ''').fetchall()
    conn.close()
    return render_template("daftar_folder.html", folders=folders)

@app.route("/folder/<int:folder_id>/edit", methods=["POST"])
@login_required
def edit_folder(folder_id):
    nama_folder = request.form["nama_folder"]
    informasi = request.form["informasi"]

    conn = get_db_connection()
    conn.execute(
        "UPDATE folder SET nama_folder = ?, informasi = ? WHERE id = ?",
        (nama_folder, informasi, folder_id),
    )
    conn.commit()
    conn.close()

    flash("Folder updated successfully!", "success")
    return redirect(url_for("daftar_folder"))

@app.route("/tambah_folder", methods=["POST"])
@login_required
def tambah_folder():
    nama_folder = request.form["nama_folder"]
    informasi = request.form.get("informasi", "")
    tanggal = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    conn.execute("INSERT INTO folder (nama_folder, informasi, tanggal_dibuat) VALUES (?, ?, ?)",
                 (nama_folder, informasi, tanggal))
    conn.commit()
    conn.close()

    return redirect(url_for("daftar_folder"))

@app.route("/folder/<int:folder_id>")
@login_required
def lihat_foto_folder(folder_id):
    conn = get_db_connection()

    # Ambil info folder
    folder = conn.execute("SELECT * FROM folder WHERE id = ?", (folder_id,)).fetchone()

    # Ambil daftar hasil analisa (tanpa koordinat dulu)
    hasil_rows = conn.execute("""
        SELECT ha.*, GROUP_CONCAT(hdk.jenis_kerusakan || ' (' || hdk.tingkat_kerusakan || ')', ', ') as detail_kerusakan
        FROM hasil_analisa ha
        LEFT JOIN hasil_detail_kerusakan hdk ON ha.id = hdk.hasil_id
        WHERE ha.folder_id = ?
        GROUP BY ha.id
        ORDER BY ha.id DESC
    """, (folder_id,)).fetchall()

    hasil_list = []
    for h in hasil_rows:
        # Ambil koordinat bounding box untuk hasil ini
        boxes = conn.execute("""
            SELECT x1, y1, x2, y2, jenis_kerusakan AS jenis, tingkat_kerusakan AS tingkat_kerusakan, luas_persen, deduct_value
            FROM hasil_detail_kerusakan
            WHERE hasil_id = ?
        """, (h["id"],)).fetchall()

        boxes_json = [dict(b) for b in boxes]
        hasil_list.append(dict(h, boxes_json=boxes_json))


    conn.close()
    return render_template("detail_folder.html", folder=folder, hasil_list=hasil_list)

from ultralytics import YOLO
import cv2
import numpy as np

@app.route("/upload_analisa/<int:folder_id>", methods=["POST"])
@login_required
def upload_analisa(folder_id):
    images = request.files.getlist("image[]")  # ambil semua file

    if not images or images == [None]:
        flash("Tidak ada file yang diupload.")
        return redirect(url_for("detail_folder", folder_id=folder_id))

    conn = get_db_connection()
    folder = conn.execute("SELECT * FROM folder WHERE id = ?", (folder_id,)).fetchone()
    if not folder:
        flash("Folder tidak ditemukan.")
        return redirect(url_for("index"))

    folder_path = os.path.join("static/uploads_analisa", folder["nama_folder"])
    os.makedirs(folder_path, exist_ok=True)

    from ultralytics import YOLO
    import uuid
    from datetime import datetime
    import cv2

    model = YOLO("yolo/weights/latest.pt")
    print(f"[INFO] Model Loaded: {model.names}")
    cur = conn.cursor()

    for image in images:
        if image and image.filename:
            ext = image.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            save_path = os.path.join(folder_path, filename)
            image.save(save_path)
            print(f"[INFO] Saved: {save_path}")

            # === ANALISIS YOLO ===
            results = model(save_path, conf=0.05)[0]
            print(f"[RESULT] : {results}")
            if len(results.boxes) == 0:
                print(f"[INFO] Tidak ada kerusakan terdeteksi pada {filename}")
                continue
            print(f"[INFO] Deteksi: {filename} - {len(results.boxes)} box")

            img = cv2.imread(save_path)
            img_h, img_w = img.shape[:2]
            
            kerusakan_data = []
            total_mask = np.zeros((img_h, img_w), dtype=np.uint8)  # mask kosong

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                jenis = model.names[cls_id]

                # Warnai area bounding box di mask
                total_mask[y1:y2, x1:x2] = 1

                # Simpan box info dulu (luas dihitung nanti)
                kerusakan_data.append({
                    "jenis": jenis,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
             
            # Hitung total luas area kerusakan unik
            total_area = np.sum(total_mask)
            total_area_persen = (total_area / (img_w * img_h)) * 100

            final_kerusakan_data = []

            for item in kerusakan_data:
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                box_mask = np.zeros_like(total_mask)
                box_mask[y1:y2, x1:x2] = 1
                overlap = np.logical_and(box_mask, total_mask)
                overlap_area = np.sum(overlap)

                luas_persen = (overlap_area / (img_w * img_h)) * 100
                tingkat = tentukan_tingkat(luas_persen)

                final_kerusakan_data.append({
                    "jenis": item["jenis"],
                    "tingkat": tingkat,
                    "luas": luas_persen,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            # === Perbaikan total_deduct pakai proporsi dari total_area_persen ===
            total_luas_box = sum([item["luas"] for item in final_kerusakan_data])
            total_deduct = 0
            for item in final_kerusakan_data:
                if total_luas_box == 0:
                    proporsi = 0
                else:
                    proporsi = item["luas"] / total_luas_box

                luas_distribusi = proporsi * total_area_persen
                item["luas_terkoreksi"] = luas_distribusi
                dv = hitung_dv(item["jenis"], item["tingkat"], luas_distribusi)
                item["dv"] = dv  # pakai nilai terkoreksi
                total_deduct += dv

            if total_deduct > 100:
                scale = 100 / total_deduct
                for item in final_kerusakan_data:
                    item["dv"] *= scale
                total_deduct = 100

            nilai_pci = max(0, 100 - total_deduct)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("""
                INSERT INTO hasil_analisa (folder_id, filename, nilai_pci, tanggal_analisa, luas_total, total_deduct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (folder_id, filename, nilai_pci, now, total_area_persen, total_deduct))
            hasil_id = cur.lastrowid

            for item in final_kerusakan_data:
                cur.execute("""
                    INSERT INTO hasil_detail_kerusakan 
                    (hasil_id, jenis_kerusakan, tingkat_kerusakan, luas_persen, deduct_value, x1, y1, x2, y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hasil_id,
                    item["jenis"],
                    item["tingkat"],
                    item["luas"],
                    item["dv"],
                    item["x1"],
                    item["y1"],
                    item["x2"],
                    item["y2"]
                ))

            print(f"[INFO] {filename} selesai, PCI: {nilai_pci:.2f}")

    conn.commit()
    conn.close()

    # flash(f"{len(images)} gambar berhasil dianalisis.")
    return redirect(url_for("lihat_foto_folder", folder_id=folder_id))

def hitung_dv(jenis, tingkat, luas):
    
    base_dv = {
        "Lubang": 1.0,
        "Retak Buaya": 0.8,
        "Retak Memanjang": 0.5,
        "Retak Melintang": 0.5
    }

    multiplier = {
        "Ringan": 1.0,
        "Sedang": 1.5,
        "Berat": 2.0
    }

    nilai_dv = base_dv.get(jenis, 1.0) * multiplier.get(tingkat, 1.0) * luas
    print(f"[INFO] Jenis: {jenis}, Tingkat: {tingkat}, Luas: {luas}, Nilai DV: {nilai_dv}")

    return nilai_dv

# Tambahkan fungsi ini di atas:
def tentukan_tingkat(luas):
    if luas < 10:
        tingkat = "Ringan"
    elif luas < 40:
        tingkat = "Sedang"
    else:
        tingkat = "Berat"
    return tingkat

@app.route("/add_label/<int:image_id>")
@login_required
def add_label(image_id):
    conn = get_db_connection()
    image = conn.execute("SELECT * FROM sample_foto WHERE id = ?", (image_id,)).fetchone()
    conn.close()

    if image is None:
        return "Gambar tidak ditemukan", 404

    return render_template("add_label.html", image=image)

@app.route("/delete_sample/<filename>", methods=["POST"])
@login_required
def delete_sample(filename):
    import os
    from pathlib import Path

    # Hapus file gambar dari static/uploads
    filepath = os.path.join("static", "uploads", filename)
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass

    # Ambil nama dasar file tanpa ekstensi
    base_filename = filename.rsplit(".", 1)[0]

    # Semua file yang perlu dihapus
    files_to_delete = [
        os.path.join("yolo", "labels", "train", base_filename + ".txt"),
        os.path.join("yolo", "dataset", "images", "train", filename),
        os.path.join("yolo", "dataset", "labels", "train", base_filename + ".txt"),
        os.path.join("yolo", "dataset", "labels", "train", base_filename + ".json"),  # JSON metadata
    ]

    # Hapus semua file tersebut
    for path in files_to_delete:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Gagal menghapus {path}: {e}")

    # Hapus dari database
    conn = get_db_connection()
    conn.execute("DELETE FROM sample_foto WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

    return redirect(url_for("labeling"))

@app.route('/delete_hasil/<int:hasil_id>', methods=['POST'])
@login_required
def delete_hasil(hasil_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM hasil_analisa WHERE id = ?", (hasil_id,)).fetchone()

    if row:
        # Hapus file fisik
        folder_row = conn.execute("SELECT nama_folder FROM folder WHERE id = ?", (row['folder_id'],)).fetchone()
        if folder_row:
            path = os.path.join("static", "uploads_analisa", folder_row['nama_folder'], row['filename'])
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        # Hapus dari DB
        conn.execute("DELETE FROM hasil_detail_kerusakan WHERE hasil_id = ?", (hasil_id,))
        conn.execute("DELETE FROM hasil_analisa WHERE id = ?", (hasil_id,))
        conn.commit()

    conn.close()

    return jsonify({"success": True})
    
@app.route('/train_yolo', methods=['POST'])
@login_required
def train_yolo():
    import subprocess
    import datetime
    import sys
    import os
    import shutil
    import time

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'yolov5_{timestamp}'
    script_path = os.path.join(os.path.dirname(__file__), "yolo", "train.py")

    # Jalankan proses training secara sinkron
    process = subprocess.run([
        sys.executable, script_path,
        '--img', '640',
        '--batch', '16',
        '--epochs', '10',
        '--data', 'yolo/data.yaml',
        '--weights', 'yolov5s.pt',
        '--name', model_name
    ])

    # Cek apakah training sukses
    trained_model_path = os.path.join("runs", "detect", model_name, "weights", "best.pt")
    if os.path.exists(trained_model_path):
        # Pastikan folder weights ada
        os.makedirs("yolo/weights", exist_ok=True)
        shutil.copy(trained_model_path, "yolo/weights/latest.pt")
        print(f"[INFO] Model berhasil disalin ke yolo/weights/latest.pt")
    else:
        print(f"[ERROR] Model tidak ditemukan di: {trained_model_path}")

    return redirect(url_for('training_page'))

@app.route('/training')
@login_required
def training_page():
    weights_dir = 'yolo/weights'
    latest_model = None
    if os.path.exists(weights_dir):
        models = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
        if models:
            latest_model = max(models)
    return render_template('training.html', latest_model=latest_model)

import os
import json
from flask import request, jsonify
from PIL import Image

@app.route("/save_label/<int:image_id>", methods=["POST"])
@login_required
def save_label(image_id):
    from PIL import Image
    import os, json, shutil

    data = request.get_json()
    boxes = data.get("boxes", [])

    if not boxes:
        return jsonify({"error": "Tidak ada data kotak yang diterima"}), 400

    conn = get_db_connection()
    row = conn.execute("SELECT filename FROM sample_foto WHERE id = ?", (image_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Gambar tidak ditemukan"}), 404

    filename = row["filename"]
    conn.close()

    image_path = os.path.join("static", "uploads", filename)
    label_path = os.path.join("yolo", "dataset", "labels", "train", filename.replace(".jpg", ".txt"))
    json_path = label_path.replace(".txt", ".json")

    if not os.path.exists(image_path):
        return jsonify({"error": "File gambar tidak ditemukan"}), 404

    dst_image_path = os.path.join("yolo", "dataset", "images", "train", filename)
    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
    shutil.copyfile(image_path, dst_image_path)

    img = Image.open(image_path)
    W, H = img.size

    def get_class_id(jenis):
        mapping = {
            "Lubang": 0,
            "Retak Buaya": 1,
            "Retak Memanjang": 2,
            "Retak Melintang": 3
        }
        return mapping.get(jenis, 0)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    try:
        with open(label_path, "w") as f:
            for box in boxes:
                x1, y1, x2, y2, jenis = box
                x_center = (x1 + x2) / 2 / W
                y_center = (y1 + y2) / 2 / H
                width = abs(x2 - x1) / W
                height = abs(y2 - y1) / H
                class_id = get_class_id(jenis)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan label: {str(e)}"}), 500

    try:
        metadata = []
        for box in boxes:
            x1, y1, x2, y2, jenis = box
            metadata.append({
                "jenis": jenis,
                "coords": [x1, y1, x2, y2]
            })
        with open(json_path, "w") as jf:
            json.dump(metadata, jf, indent=2)
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan metadata JSON: {str(e)}"}), 500

    print("DATA BOXES.. :", boxes)

    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM kerusakan_detail WHERE sample_id = ?", (image_id,))
        for box in boxes:
            x1, y1, x2, y2, jenis = box
            conn.execute('''
                INSERT INTO kerusakan_detail (sample_id, jenis, x1, y1, x2, y2)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_id, jenis, int(x1), int(y1), int(x2), int(y2)))
        conn.execute("UPDATE sample_foto SET labeled = 1 WHERE id = ?", (image_id,))

        cursor = conn.execute('''
            SELECT jenis FROM kerusakan_detail WHERE sample_id = ?
        ''', (image_id,))
        label_data = cursor.fetchall()
        ringkasan = ", ".join([jenis for (jenis,) in label_data])

        conn.execute('''
            UPDATE sample_foto SET jenis_kerusakan = ? WHERE id = ?
        ''', (ringkasan, image_id))

        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan ke database: {str(e)}"}), 500

def get_class_id(jenis):
    mapping = {
        "Lubang": 0,
        "Retak Buaya": 1,
        "Retak Memanjang": 2,
        "Retak Melintang": 3
    }
    return mapping.get(jenis, 0)

@app.route("/delete_bulk", methods=["POST"])
@login_required
def delete_bulk():
    data = request.get_json()
    ids = data.get("ids", [])

    if not ids:
        return jsonify({"success": False, "message": "No IDs provided."})

    conn = get_db_connection()
    cur = conn.cursor()
    for hasil_id in ids:
        # Hapus dari detail dulu
        cur.execute("DELETE FROM hasil_detail_kerusakan WHERE hasil_id = ?", (hasil_id,))
        cur.execute("DELETE FROM hasil_analisa WHERE id = ?", (hasil_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True})

@app.route("/users")
@login_required
@admin_required
def list_users():
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, role FROM users ORDER BY username").fetchall()
    conn.close()
    return render_template("users.html", users=users)

@app.route("/add_user", methods=["POST"])
@login_required
@admin_required
def add_user():
    username = request.form["username"]
    password = request.form["password"]
    role = request.form["role"]

    if not password:
        flash("Password is required.", "danger")
        return redirect(url_for('list_users'))

    conn = get_db_connection()
    # Cek apakah username sudah ada
    user_exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()

    if user_exists:
        flash(f"Username '{username}' already exists. Please choose a different one.", "danger")
        conn.close()
        return redirect(url_for('list_users'))

    # Hash password
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    # Insert user baru
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                 (username, hashed_password, role))
    conn.commit()
    conn.close()

    flash(f"User '{username}' has been added successfully.", "success")
    return redirect(url_for('list_users'))

@app.route("/edit_user/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def edit_user(user_id):
    username = request.form["username"]
    password = request.form.get("password")
    role = request.form["role"]

    conn = get_db_connection()

    # Mencegah perubahan role pada admin utama (ID 1)
    if user_id == 1 and role != 'admin':
        flash("The role of the primary admin user cannot be changed.", "danger")
        conn.close()
        return redirect(url_for('list_users'))

    # Cek apakah username baru sudah digunakan oleh user lain
    user_exists = conn.execute(
        "SELECT id FROM users WHERE username = ? AND id != ?", (username, user_id)
    ).fetchone()

    if user_exists:
        flash(f"Username '{username}' is already taken. Please choose another.", "danger")
        conn.close()
        return redirect(url_for('list_users'))

    if password:
        # Jika password baru diisi, hash dan update
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        conn.execute(
            "UPDATE users SET username = ?, password_hash = ?, role = ? WHERE id = ?",
            (username, hashed_password, role, user_id)
        )
    else:
        # Jika password kosong, update field lain tanpa mengubah password
        conn.execute(
            "UPDATE users SET username = ?, role = ? WHERE id = ?",
            (username, role, user_id)
        )

    conn.commit()
    conn.close()

    flash(f"User '{username}' has been updated successfully.", "success")
    return redirect(url_for('list_users'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == "__main__":
    ensure_columns_exist() 
    app.run(debug=True)
