"""
rppgforensik_v2.py  —  UAS Forensik Digital
=====================================================================
Pengukuran detak jantung non-kontak (rPPG, algoritma POS) + Liveness
Detection, dengan PENCATATAN DATA LENGKAP sebagai bukti ilmiah.

Cara jalan (PyCharm -> Run, atau terminal):
    D:\\PycharmProjects\\.venv\\Scripts\\python.exe rppgforensik_v2.py --subjek S01
    ... --subjek S02 --fitzpatrick 4          (opsional: tipe kulit manual 1-6)
    ... --video D:\\data\\subject1\\vid.avi    (mode offline / video dataset)

Tombol:
    [1] Live   [2] Live+Makeup   [3] Foto Cetak   [4] Layar HP   (pilih SEBELUM mengukur)
    [R]/[SPASI] ukur ulang     [E] kunci/lepas eksposur kamera     [Q]/[ESC] keluar
    Setelah hasil terkunci: ketik angka BPM oksimeter lalu [ENTER], atau [N] untuk lewati.

Output per sesi  ->  hasil_sesi/<waktu>_<subjek>_<kondisi>/
    raw_signal.csv     : data mentah setiap frame (waktu, RGB tiap ROI, posisi hidung)
    processed.csv      : sinyal setiap tahap (RGB 30 Hz, Cn, S1, S2, POS, filter)
    spectrum.csv       : spektrum daya (frekuensi, BPM, daya)
    pipeline.png       : grafik 6 tahap perubahan sinyal dari awal sampai akhir
    snapshot.jpg       : foto wajah + ROI saat terkunci
    ringkasan.json     : semua angka hasil
Rekap semua sesi    ->  data_uji_rppg_v2.csv   (dipakai analisis_validasi.py)
=====================================================================
"""
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import csv
import json
import time
import argparse
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import rppg_core as rc

# =====================================================================
# A. ARGUMEN & KONFIGURASI
# =====================================================================
ap = argparse.ArgumentParser()
ap.add_argument("--subjek", default="S01", help="ID subjek, mis. S01")
ap.add_argument("--fitzpatrick", type=int, default=0, help="Tipe kulit Fitzpatrick manual 1-6 (0 = tidak diisi)")
ap.add_argument("--kamera", type=int, default=0)
ap.add_argument("--video", default="", help="Path video (mode offline). Kosong = webcam")
ap.add_argument("--ref", type=float, default=0.0, help="BPM referensi (untuk mode video dataset)")
ap.add_argument("--durasi", type=float, default=20.0, help="Lama pengukuran (detik)")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--no-lock", action="store_true", help="Jangan kunci eksposur/white-balance")
ap.add_argument("--rekam-layar", action="store_true", help="Rekam dashboard ke MP4 per sesi")
args = ap.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
OUT_ROOT = os.path.join(BASE_DIR, "hasil_sesi")
MASTER_CSV = os.path.join(BASE_DIR, "data_uji_rppg_v2.csv")
os.makedirs(OUT_ROOT, exist_ok=True)

WARMUP_SEC = 3.0           # kalibrasi: eksposur stabil + ukur warna kulit (ITA)
MEASURE_SEC = args.durasi  # durasi pengukuran tetap (protokol yang sama tiap subjek)
MIN_ANALYZE_SEC = 6.0      # estimasi sementara mulai ditampilkan setelah 6 detik
ANALYZE_EVERY = 0.5        # hitung ulang tiap 0.5 detik (hemat CPU)
FACE_LOST_TOL = 0.7        # wajah hilang > 0.7 detik -> pengukuran diulang
MOTION_THR = 0.012         # std posisi hidung > 1.2% lebar frame = bergerak
SNR_MIN_RELIABLE = -5.0    # di bawah ini BPM dianggap TIDAK ANDAL

KONDISI = {ord('1'): ("live", "Wajah Asli"),
           ord('2'): ("makeup", "Wajah Asli + Makeup"),
           ord('3'): ("foto", "Foto Cetak (Spoof)"),
           ord('4'): ("layar", "Layar HP (Spoof)")}
kondisi_kode, kondisi_nama = "live", "Wajah Asli"

# ROI wajah (indeks MediaPipe Face Mesh 478 titik) dan bobotnya
ROI = {"dahi": [109, 338, 337, 336, 107, 108],
       "pipi_kiri": [117, 118, 101, 205, 187, 123],
       "pipi_kanan": [346, 347, 330, 425, 411, 352]}
BOBOT = {"dahi": 0.50, "pipi_kiri": 0.25, "pipi_kanan": 0.25}

# =====================================================================
# B. MODEL MEDIAPIPE (Tasks API, mode VIDEO -> pelacakan antar-frame)
# =====================================================================
if not os.path.exists(MODEL_PATH):
    print("Mengunduh face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1))

# =====================================================================
# C. KAMERA / VIDEO
# =====================================================================
IS_VIDEO = bool(args.video)
if IS_VIDEO:
    cap = cv2.VideoCapture(args.video)
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
else:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.kamera, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

exposure_locked = False


def lock_camera(lock):
    """
    Auto-exposure & auto-white-balance webcam terus mengubah kecerahan/warna
    -> ini noise terbesar rPPG. Setelah kalibrasi, nilai dikunci.
    (DirectShow: 0.25 = manual, 0.75 = auto. Tidak semua webcam mendukung.)
    """
    global exposure_locked
    if IS_VIDEO or args.no_lock:
        return
    if lock:
        exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        exposure_locked = True
        print(f"[KAMERA] Eksposur dikunci pada {exp:.1f}, auto-WB dimatikan.")
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        exposure_locked = False
        print("[KAMERA] Eksposur & WB kembali otomatis.")


# =====================================================================
# D. EKSTRAKSI ROI (rata-rata RGB piksel kulit di dalam poligon)
# =====================================================================
def roi_mean_rgb(frame, lm, idx, w, h):
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idx], dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x, y = max(x, 0), max(y, 0)
    crop = frame[y:y + bh, x:x + bw]
    if crop.size == 0:
        return None, pts, None
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts - [x, y]], 255)
    # buang piksel kilap (overexposed) & terlalu gelap (bayangan/alis)
    mx, mn = crop.max(axis=2), crop.min(axis=2)
    mask[(mx >= 250) | (mn <= 15)] = 0
    if cv2.countNonZero(mask) < 30:
        return None, pts, None
    b, g, r, _ = cv2.mean(crop, mask=mask)
    return np.array([r, g, b]), pts, crop[mask > 0]


# =====================================================================
# E. STATE PENGUKURAN
# =====================================================================
class Sesi:
    def __init__(self):
        self.fase = "MENUNGGU_WAJAH"
        self.t0_warm = None
        self.t, self.rgb_w, self.nose, self.moving = [], [], [], []
        self.rois = {k: [] for k in ROI}
        self.skin_px = []
        self.ita = None
        self.res = None
        self.last_analyze = 0.0
        self.last_face = None
        self.final = None
        self.live = None
        self.alasan = {}
        self.folder = None
        self.input_ref = ""
        self.saved_master = False
        self.snapshot = None
        self.writer = None


S = Sesi()


def reset(msg="Pengukuran diulang"):
    global S
    if S.final is not None and not S.saved_master:
        finalize(None)  # simpan dulu data yang sudah terkunci agar tidak hilang
    if S.writer is not None:
        S.writer.release()
    S = Sesi()
    print(f"[INFO] {msg}")


# =====================================================================
# F. PENYIMPANAN DATA (BUKTI)
# =====================================================================
def simpan_sesi():
    """Dipanggil saat hasil terkunci: simpan seluruh data mentah & perantara."""
    tag = time.strftime("%Y%m%d_%H%M%S")
    S.folder = os.path.join(OUT_ROOT, f"{tag}_{args.subjek}_{kondisi_kode}")
    os.makedirs(S.folder, exist_ok=True)
    r = S.final

    with open(os.path.join(S.folder, "raw_signal.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_detik", "R_w", "G_w", "B_w",
                     "R_dahi", "G_dahi", "B_dahi", "R_kiri", "G_kiri", "B_kiri",
                     "R_kanan", "G_kanan", "B_kanan", "hidung_x", "hidung_y", "bergerak"])
        t0 = S.t[0]
        for i in range(len(S.t)):
            wr.writerow([f"{S.t[i] - t0:.4f}", *np.round(S.rgb_w[i], 4),
                         *np.round(S.rois["dahi"][i], 4), *np.round(S.rois["pipi_kiri"][i], 4),
                         *np.round(S.rois["pipi_kanan"][i], 4),
                         f"{S.nose[i][0]:.1f}", f"{S.nose[i][1]:.1f}", int(S.moving[i])])

    with open(os.path.join(S.folder, "processed.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_detik", "R_30Hz", "G_30Hz", "B_30Hz", "Rn", "Gn", "Bn",
                     "S1", "S2", "alpha", "POS_mentah", "POS_filter", "puncak"])
        pk = set(r["peaks"].tolist())
        tu = r["t_uniform"] - r["t_uniform"][0]
        for i in range(len(tu)):
            wr.writerow([f"{tu[i]:.4f}", *np.round(r["rgb_uniform"][i], 4), *np.round(r["rgb_norm"][i], 6),
                         f"{r['S1'][i]:.6f}", f"{r['S2'][i]:.6f}", f"{r['alpha'][i]:.5f}",
                         f"{r['pos_raw'][i]:.6f}", f"{r['pos_filtered'][i]:.6f}", int(i in pk)])

    with open(os.path.join(S.folder, "spectrum.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frekuensi_Hz", "BPM", "daya_PSD"])
        m = r["freqs"] <= 4.0
        for fr, p in zip(r["freqs"][m], r["psd"][m]):
            wr.writerow([f"{fr:.5f}", f"{fr * 60:.3f}", f"{p:.6e}"])

    if S.snapshot is not None:
        cv2.imwrite(os.path.join(S.folder, "snapshot.jpg"), S.snapshot)
    tulis_ringkasan(None)
    rc.plot_pipeline(r, os.path.join(S.folder, "pipeline.png"),
                     judul=f"{args.subjek} | {kondisi_nama} | rPPG {r['bpm']:.1f} BPM")
    print(f"[DATA] Sesi disimpan di {S.folder}")


def ringkasan_dict(ref):
    r = S.final
    ita = S.ita or (None, None, None, None, "-")
    err = (r["bpm"] - ref) if ref else None
    motion_pct = 100.0 * np.mean(S.moving) if S.moving else 0.0
    return {
        "Waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Subjek": args.subjek, "Kondisi": kondisi_kode,
        "Label_Asli": "live" if kondisi_kode in ("live", "makeup") else "spoof",
        "Fitzpatrick_manual": args.fitzpatrick or "",
        "ITA_derajat": round(ita[0], 2) if ita[0] is not None else "",
        "L_star": round(ita[1], 2) if ita[1] is not None else "",
        "b_star": round(ita[3], 2) if ita[3] is not None else "",
        "Kategori_Kulit_ITA": ita[4],
        "Durasi_detik": round(S.t[-1] - S.t[0], 2),
        "FPS_efektif": round(r["fs_eff"], 2),
        "Jumlah_Frame": len(S.t),
        "Persen_Frame_Bergerak": round(motion_pct, 1),
        "BPM_rPPG_POS": round(r["bpm"], 2),
        "BPM_IBI": round(r["bpm_ibi"], 2) if r["bpm_ibi"] else "",
        "BPM_CHROM": round(r["bpm_chrom"], 2),
        "BPM_GREEN": round(r["bpm_green"], 2),
        "BPM_Oksimeter": ref if ref else "",
        "Error_BPM": round(err, 2) if err is not None else "",
        "Abs_Error_BPM": round(abs(err), 2) if err is not None else "",
        "SNR_dB": round(r["snr_db"], 3),
        "Korelasi_ROI": round(r["roi_corr"], 3) if r["roi_corr"] is not None else "",
        "BPM_Dahi": round(r["roi_bpm"].get("dahi", 0), 2),
        "BPM_PipiKiri": round(r["roi_bpm"].get("pipi_kiri", 0), 2),
        "BPM_PipiKanan": round(r["roi_bpm"].get("pipi_kanan", 0), 2),
        "Selisih_BPM_ROI": round(r["roi_bpm_spread"], 2) if r["roi_bpm_spread"] is not None else "",
        "Prediksi_Liveness": "live" if S.live else "spoof",
        "Keandalan_BPM": "andal" if r["snr_db"] >= SNR_MIN_RELIABLE else "tidak_andal",
        "Eksposur_Dikunci": exposure_locked,
        "Folder": os.path.relpath(S.folder, BASE_DIR) if S.folder else "",
    }


def tulis_ringkasan(ref):
    with open(os.path.join(S.folder, "ringkasan.json"), "w", encoding="utf-8") as f:
        json.dump(ringkasan_dict(ref), f, indent=2, ensure_ascii=False, default=str)


def finalize(ref):
    """Tulis 1 baris ke CSV rekap (dipanggil setelah BPM oksimeter diisi / dilewati)."""
    if S.saved_master or S.final is None:
        return
    row = ringkasan_dict(ref)
    baru = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        if baru:
            wr.writeheader()
        wr.writerow(row)
    if ref:
        tulis_ringkasan(ref)
        rc.plot_pipeline(S.final, os.path.join(S.folder, "pipeline.png"),
                         judul=f"{args.subjek} | {kondisi_nama} | rPPG {S.final['bpm']:.1f} vs Oksimeter {ref:.0f} BPM",
                         bpm_ref=ref)
    S.saved_master = True
    print(f"[REKAP] {row['Subjek']} {row['Kondisi']} | rPPG {row['BPM_rPPG_POS']} | "
          f"Oksimeter {row['BPM_Oksimeter'] or '-'} | Error {row['Error_BPM'] or '-'} | {row['Prediksi_Liveness']}")


# =====================================================================
# G. FUNGSI GAMBAR DASHBOARD
# =====================================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX


def txt(img, s, x, y, sc=0.45, col=(230, 230, 230), th=1):
    cv2.putText(img, s, (x, y), FONT, sc, col, th, cv2.LINE_AA)


def plot_box(img, data, x, y, w, h, judul, col=(0, 255, 120), marks=None, xr=None, vline=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), (28, 28, 28), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    txt(img, judul, x + 6, y + 15, 0.42, (200, 255, 200))
    if data is None or len(data) < 2:
        txt(img, "menunggu data...", x + 6, y + h // 2, 0.4, (120, 120, 120))
        return
    d = np.asarray(data, float)
    lo, hi = np.min(d), np.max(d)
    rng = hi - lo if hi - lo > 1e-9 else 1.0
    px = x + 6 + (np.arange(len(d)) / (len(d) - 1) * (w - 12)).astype(int)
    py = y + h - 6 - ((d - lo) / rng * (h - 28)).astype(int)
    cv2.polylines(img, [np.column_stack([px, py]).astype(np.int32)], False, col, 1, cv2.LINE_AA)
    if marks is not None:
        for m in marks:
            if 0 <= m < len(d):
                cv2.circle(img, (int(px[m]), int(py[m])), 3, (0, 0, 255), -1)
    if xr is not None:  # sumbu-x spektrum dalam BPM
        for b in (60, 90, 120, 150):
            xx = x + 6 + int((b - xr[0]) / (xr[1] - xr[0]) * (w - 12))
            cv2.line(img, (xx, y + h - 4), (xx, y + h - 10), (150, 150, 150), 1)
            txt(img, str(b), xx - 9, y + h - 12, 0.33, (150, 150, 150))
        if vline is not None:
            xx = x + 6 + int((vline - xr[0]) / (xr[1] - xr[0]) * (w - 12))
            cv2.line(img, (xx, y + 20), (xx, y + h - 4), (0, 0, 255), 1)


def draw_dashboard(frame, now):
    W, H = 1280, 720
    can = np.zeros((H, W, 3), np.uint8)
    cam = cv2.resize(frame, (800, 450))
    can[70:520, 0:800] = cam

    # ---------- HEADER ----------
    tahap = {"MENUNGGU_WAJAH": "1. Deteksi wajah", "KALIBRASI": "2. Kalibrasi cahaya & warna kulit",
             "MENGUKUR": "3. Pengumpulan sinyal", "TERKUNCI": "4. Hasil terkunci - isi BPM oksimeter",
             "TERSIMPAN": "5. Data tersimpan"}[S.fase]
    txt(can, f"rPPG Forensik v2 | Subjek: {args.subjek} | Kondisi: {kondisi_nama} | Tahap: {tahap}", 10, 22, 0.55, (0, 255, 255), 1)
    txt(can, "[1]Live [2]Makeup [3]Foto [4]Layar  [R]Ulang  [E]Kunci eksposur  [Q]Keluar"
        f"   | Eksposur: {'TERKUNCI' if exposure_locked else 'auto'}", 10, 44, 0.45, (190, 190, 190))
    el = (S.t[-1] - S.t[0]) if len(S.t) > 1 else 0.0
    prog = min(1.0, el / MEASURE_SEC) if S.fase in ("MENGUKUR", "TERKUNCI", "TERSIMPAN") else 0.0
    cv2.rectangle(can, (10, 54), (790, 64), (60, 60, 60), -1)
    cv2.rectangle(can, (10, 54), (10 + int(780 * prog), 64), (0, 200, 100), -1)
    txt(can, f"{el:4.1f}/{MEASURE_SEC:.0f} s", 800, 64, 0.45)

    # ---------- PANEL KANAN: 4 GRAFIK TAHAPAN SINYAL ----------
    x0, wbox, hbox = 810, 460, 150
    g_raw = [v[1] for v in S.rgb_w[-300:]] if S.rgb_w else None
    plot_box(can, g_raw, x0, 75, wbox, hbox, "Tahap 1: Kanal G mentah (rata2 ROI) - 10 s terakhir", (80, 255, 80))
    r = S.final if S.final is not None else S.res
    if r is not None:
        n = int(10 * rc.TARGET_FS)
        plot_box(can, r["pos_raw"][-n:], x0, 235, wbox, hbox, "Tahap 2: Sinyal POS  H = S1 + alpha*S2", (255, 120, 255))
        pf = r["pos_filtered"][-n:]
        off = len(r["pos_filtered"]) - len(pf)
        mk = [p - off for p in r["peaks"] if p >= off]
        plot_box(can, pf, x0, 395, wbox, hbox, "Tahap 3: Band-pass 0.75-3 Hz + puncak denyut (IBI)", (255, 255, 255), marks=mk)
        f, p = r["freqs"], r["psd"]
        m = (f >= 0.75) & (f <= 3.0)
        plot_box(can, p[m], x0, 555, wbox, 160, f"Tahap 4: Spektrum Welch -> puncak {r['bpm']:.1f} BPM",
                 (0, 170, 255), xr=(45, 180), vline=r["bpm"])
    else:
        for yy, jj in [(235, "Tahap 2: Sinyal POS"), (395, "Tahap 3: Band-pass + puncak"), (555, "Tahap 4: Spektrum")]:
            plot_box(can, None, x0, yy, wbox, hbox, jj)

    # ---------- PANEL BAWAH: DASAR PERHITUNGAN (ANGKA PERANTARA) ----------
    y = 540
    if S.rgb_w:
        R, G, B = S.rgb_w[-1]
        txt(can, f"Frame: rata2 RGB ROI berbobot = (0.5 dahi + 0.25 pipiKi + 0.25 pipiKa) -> R={R:.2f} G={G:.2f} B={B:.2f}", 10, y); y += 20
    if S.ita:
        txt(can, f"Warna kulit: ITA = arctan((L*-50)/b*) = arctan(({S.ita[1]:.1f}-50)/{S.ita[3]:.1f}) = {S.ita[0]:.1f} deg -> {S.ita[4]}", 10, y, 0.45, (255, 220, 150)); y += 20
    if r is not None:
        mu = r["rgb_mean"]
        alpha_last = r["alpha"][r["alpha"] > 0][-1] if np.any(r["alpha"] > 0) else 0
        txt(can, f"FPS efektif={r['fs_eff']:.1f} -> interpolasi 30 Hz | N={len(r['t_uniform'])} sampel | mean RGB=({mu[0]:.1f},{mu[1]:.1f},{mu[2]:.1f})", 10, y); y += 20
        txt(can, f"POS: Cn=C/mean(C); S1=Gn-Bn; S2=-2Rn+Gn+Bn; alpha=std(S1)/std(S2)={alpha_last:.3f}", 10, y); y += 20
        ibi_s = f"{r['bpm_ibi']:.1f}" if r["bpm_ibi"] else "-"
        txt(can, f"BPM = f_puncak x 60 = {r['bpm'] / 60:.4f} Hz x 60 = {r['bpm']:.1f} | IBI: 60/median(IBI) = {ibi_s} | CHROM {r['bpm_chrom']:.1f} | GREEN {r['bpm_green']:.1f}", 10, y, 0.45, (0, 255, 255)); y += 20
        rb = r["roi_bpm"]
        corr_s = f"{r['roi_corr']:.2f}" if r["roi_corr"] is not None else "-"
        txt(can, f"Forensik: SNR={r['snr_db']:.2f} dB | korelasi ROI={corr_s} | BPM dahi/kiri/kanan = "
            f"{rb.get('dahi', 0):.0f}/{rb.get('pipi_kiri', 0):.0f}/{rb.get('pipi_kanan', 0):.0f}", 10, y); y += 20

    # ---------- OVERLAY STATUS DI ATAS KAMERA ----------
    if S.fase == "MENUNGGU_WAJAH":
        txt(can, "Arahkan wajah ke kamera...", 20, 110, 0.8, (0, 200, 255), 2)
    elif S.fase == "KALIBRASI":
        sisa = WARMUP_SEC - (now - S.t0_warm)
        txt(can, f"Kalibrasi {sisa:.1f} s - diam, cahaya stabil", 20, 110, 0.8, (0, 200, 255), 2)
    elif S.fase == "MENGUKUR":
        if S.res is not None:
            txt(can, f"Mengukur... ~{S.res['bpm']:.0f} BPM", 20, 110, 0.9, (0, 255, 255), 2)
        else:
            txt(can, f"Mengumpulkan sinyal (min {MIN_ANALYZE_SEC:.0f} s)...", 20, 110, 0.8, (0, 255, 255), 2)
        if S.moving and S.moving[-1]:
            txt(can, "PERINGATAN: kepala bergerak! Tahan posisi.", 20, 505, 0.7, (0, 0, 255), 2)
    elif S.fase in ("TERKUNCI", "TERSIMPAN") and S.final is not None:
        fr = S.final
        cv2.rectangle(can, (10, 80), (790, 215), (20, 45, 20), -1)
        andal = fr["snr_db"] >= SNR_MIN_RELIABLE
        txt(can, f"HASIL: {fr['bpm']:.0f} BPM" + ("" if andal else "  (SINYAL LEMAH - TIDAK ANDAL)"),
            20, 120, 1.0, (0, 255, 255) if andal else (0, 140, 255), 2)
        col = (0, 255, 100) if S.live else (0, 80, 255)
        txt(can, "LIVENESS: " + ("WAJAH ASLI (denyut terverifikasi)" if S.live else "INDIKASI SPOOF"), 20, 155, 0.7, col, 2)
        txt(can, "  ".join(f"{k}:{'OK' if v else 'GAGAL'}" for k, v in S.alasan.items()), 20, 180, 0.45)
        if S.fase == "TERKUNCI":
            txt(can, f"BPM Oksimeter: {S.input_ref}_   (ketik angka, ENTER simpan, N lewati)", 20, 205, 0.55, (255, 255, 0))
        else:
            txt(can, "Tersimpan. [R] subjek/kondisi berikutnya.", 20, 205, 0.55, (180, 255, 180))
    return can


# =====================================================================
# H. LOOP UTAMA
# =====================================================================
WIN = "UAS Forensik Digital - rPPG v2"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)
frame_idx, last_ts_ms = 0, -1
video_done, auto_lock_done = False, False
now, vis = 0.0, np.zeros((720, 1280, 3), np.uint8)
print("Siap. Pilih kondisi [1-4], hadapkan wajah ke kamera.")

while True:
    ok, frame = (False, None) if video_done else cap.read()
    if not ok:
        if not IS_VIDEO:
            print("Frame kamera tidak terbaca.")
            break
        if not video_done and S.fase == "MENGUKUR" and len(S.t) > 2 and S.t[-1] - S.t[0] >= MIN_ANALYZE_SEC:
            MEASURE_SEC = S.t[-1] - S.t[0]   # video habis -> analisis data yang ada
        elif S.fase not in ("MENGUKUR", "TERKUNCI", "TERSIMPAN"):
            print("Video selesai.")
            break
        video_done = True
    else:
        frame_idx += 1
        # --- waktu frame (detik). Webcam: jam presisi; video: stempel waktu file
        if IS_VIDEO:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            now = pos if pos > 0 else frame_idx / VIDEO_FPS
        else:
            now = time.perf_counter()
            frame = cv2.flip(frame, 1)

    if ok:
        h, w = frame.shape[:2]
        vis = frame.copy()
        small = cv2.cvtColor(cv2.resize(frame, (640, int(640 * h / w))), cv2.COLOR_BGR2RGB)
        ts_ms = max(last_ts_ms + 1, int(now * 1000))
        last_ts_ms = ts_ms
        det = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=small), ts_ms)

        face_ok = False
        if det.face_landmarks:
            lm = det.face_landmarks[0]
            vals, pix = {}, []
            for name, idx in ROI.items():
                rgb, pts, px = roi_mean_rgb(frame, lm, idx, w, h)
                cv2.polylines(vis, [pts], True, (0, 255, 255) if S.fase == "TERKUNCI" else (0, 255, 0), 2)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(vis, f"{name} {int(BOBOT[name] * 100)}%", (cx - 40, cy), FONT, 0.4, (255, 255, 255), 1)
                if rgb is not None:
                    vals[name] = rgb
                    if px is not None:
                        pix.append(px[::7])
            face_ok = len(vals) == 3

        if face_ok:
            S.last_face = now
            if S.fase == "MENUNGGU_WAJAH":
                S.fase, S.t0_warm, S.skin_px = "KALIBRASI", now, []
            if S.fase == "KALIBRASI":
                # kunci eksposur di tengah kalibrasi -> sisa 1.5 s untuk stabil
                if not auto_lock_done and now - S.t0_warm >= WARMUP_SEC / 2:
                    lock_camera(True)
                    auto_lock_done = True
                if now - S.t0_warm >= WARMUP_SEC / 2:
                    S.skin_px.extend(pix)     # warna kulit diukur SETELAH eksposur terkunci
                if now - S.t0_warm >= WARMUP_SEC:
                    S.ita = rc.ita_from_bgr(np.vstack(S.skin_px)) if S.skin_px else None
                    S.fase = "MENGUKUR"
                    if S.ita:
                        print(f"[KULIT] ITA = {S.ita[0]:.1f} deg ({S.ita[4]})")
            elif S.fase == "MENGUKUR":
                nx, ny = lm[1].x * w, lm[1].y * h
                S.nose.append((nx, ny))
                mv = False
                if len(S.nose) > 15:
                    mv = np.max(np.std(np.array(S.nose[-15:]), axis=0)) > MOTION_THR * w
                S.moving.append(mv)
                S.t.append(now)
                S.rgb_w.append(sum(BOBOT[k] * vals[k] for k in ROI))
                for k in ROI:
                    S.rois[k].append(vals[k])
        else:
            if S.fase in ("KALIBRASI", "MENGUKUR") and S.last_face is not None and now - S.last_face > FACE_LOST_TOL:
                reset("Wajah hilang > 0.7 s, pengukuran diulang")

    # --- analisis berkala & penguncian
    if S.fase == "MENGUKUR" and len(S.t) > 2:
        el = S.t[-1] - S.t[0]
        if el >= MIN_ANALYZE_SEC and now - S.last_analyze >= ANALYZE_EVERY:
            S.last_analyze = now
            try:
                S.res = rc.analyze_window(np.array(S.t), np.array(S.rgb_w))
            except Exception as e:
                print("Analisis gagal:", e)
        if el >= MEASURE_SEC or (not ok):
            S.final = rc.analyze_window(np.array(S.t), np.array(S.rgb_w),
                                        {k: np.array(v) for k, v in S.rois.items()})
            if S.final is not None:
                S.live, S.alasan = rc.classify_liveness(S.final)
                S.snapshot = vis.copy()
                S.fase = "TERKUNCI"
                if args.ref:
                    S.input_ref = f"{args.ref:g}"
                simpan_sesi()

    can = draw_dashboard(vis, now)   # bila video habis, frame terakhir tetap ditampilkan
    if args.rekam_layar and S.fase in ("KALIBRASI", "MENGUKUR", "TERKUNCI"):
        if S.writer is None:
            os.makedirs(os.path.join(OUT_ROOT, "rekaman"), exist_ok=True)
            S.writer = cv2.VideoWriter(os.path.join(OUT_ROOT, "rekaman", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.subjek}.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 720))
        S.writer.write(can)
    cv2.imshow(WIN, can)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27):
        break
    if S.fase == "TERKUNCI":
        if ord('0') <= k <= ord('9') or k == ord('.'):
            S.input_ref = (S.input_ref + chr(k))[:5]
        elif k == 8:
            S.input_ref = S.input_ref[:-1]
        elif k in (13, 10):
            try:
                ref = float(S.input_ref) if S.input_ref else None
            except ValueError:
                ref = None
            finalize(ref)
            S.fase = "TERSIMPAN"
        elif k in (ord('n'), ord('N')):
            finalize(None)
            S.fase = "TERSIMPAN"
        elif k in (ord('r'), ord(' ')):
            reset()
        continue
    if k in (ord('r'), ord(' ')):
        reset()
    elif k in (ord('e'), ord('E')):
        lock_camera(not exposure_locked)
    elif k in KONDISI and S.fase in ("MENUNGGU_WAJAH", "KALIBRASI", "TERSIMPAN"):
        kondisi_kode, kondisi_nama = KONDISI[k]
        if S.fase == "TERSIMPAN":
            reset(f"Kondisi diganti: {kondisi_nama}")
        print(f"[KONDISI] {kondisi_nama}")

if S.final is not None and not S.saved_master:
    try:
        finalize(float(S.input_ref) if S.input_ref else None)
    except ValueError:
        finalize(None)
if S.writer is not None:
    S.writer.release()
cap.release()
cv2.destroyAllWindows()
landmarker.close()
