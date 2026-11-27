import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import csv
import urllib.request
from collections import deque
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# 1. PERSIAPAN MODEL MEDIAPIPE FACE LANDMARKER
# =====================================================================
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model face_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Unduhan selesai!")

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

ROI_DAHI = [109, 338, 337, 336, 107, 108]
ROI_PIPI_KIRI = [117, 118, 101, 205, 187, 123]
ROI_PIPI_KANAN = [346, 347, 330, 425, 411, 352]

# =====================================================================
# 2. PARAMETER MODE OKSIMETER (KALIBRASI -> UKUR -> KUNCI HASIL)
# =====================================================================
TARGET_FS = 30.0          # Frekuensi sampling standar (30 Hz)
WARMUP_SECONDS = 3.0      # 3 detik awal untuk menstabilkan cahaya kamera
MIN_MEASURE_SEC = 6.0     # Mulai menampilkan estimasi BPM setelah 6 detik
LOCK_SECONDS = 18.0       # Waktu maksimal sebelum hasil dikunci otomatis (seperti oksimeter)

# Rentang frekuensi diperketat (58 - 180 BPM) agar tidak tertarik noise napas 40-55 BPM
LOW_HZ = 0.97             # ~58 BPM
HIGH_HZ = 3.00            # 180 BPM

timestamps = deque()
rgb_weighted_buf = deque()
g_dahi_buf = deque()
g_kiri_buf = deque()
g_kanan_buf = deque()
nose_pos_buf = deque()
bpm_history = deque(maxlen=12)

# Variabel Pengunci Hasil Akhir (Mode Oksimeter)
is_locked = False
locked_result = None
locked_status = ""
locked_color = (0, 255, 100)
locked_duration = 0.0


def reset_measurement():
    """Mengulang pengukuran dari awal seperti menekan tombol reset di oksimeter."""
    global is_locked, locked_result, locked_status, locked_duration
    timestamps.clear()
    rgb_weighted_buf.clear()
    g_dahi_buf.clear()
    g_kiri_buf.clear()
    g_kanan_buf.clear()
    nose_pos_buf.clear()
    bpm_history.clear()
    is_locked = False
    locked_result = None
    locked_status = ""
    locked_duration = 0.0


def bandpass_filter(signal_data, fs, lowcut=0.97, highcut=3.0, order=3):
    """Menyaring sinyal pada rentang detak jantung aktif (58 - 180 BPM)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal_data)


def extract_roi_rgb(frame_bgr, landmarks, indices, w, h):
    """Mengambil nilai rata-rata RGB dari poligon wajah."""
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
        dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)
    return mean_rgb, pts


def compute_ola_pos(rgb_uniform, fs):
    """
    Algoritma Overlap-Add POS (jendela geser 1.6 detik) sesuai standar medis rPPG.
    Menghilangkan drift cahaya lambat yang membuat BPM terjebak di angka 40-60.
    """
    n = len(rgb_uniform)
    win_len = max(16, int(1.6 * fs))
    h_sig = np.zeros(n, dtype=np.float64)

    for start in range(0, n - win_len + 1):
        end = start + win_len
        win_rgb = rgb_uniform[start:end]
        mean_c = np.mean(win_rgb, axis=0) + 1e-8
        cn = win_rgb / mean_c

        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]

        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        p = s1 + alpha * s2
        p = (p - np.mean(p)) / (np.std(p) + 1e-8)
        h_sig[start:end] += p

    return detrend(h_sig)


def analyze_rppg_oximeter():
    """
    Menghitung BPM dengan gabungan Spektrum Pre-Emphasis FFT + Jarak Antar-Puncak (IBI)
    persis seperti prinsip kerja Pulse Oximeter.
    """
    if len(timestamps) < 2:
        return None

    duration = timestamps[-1] - timestamps[0]
    if duration < MIN_MEASURE_SEC:
        return None

    num_samples = int(duration * TARGET_FS)
    if num_samples < int(MIN_MEASURE_SEC * TARGET_FS):
        return None

    # 1. Interpolasi ke 30 Hz seragam
    t_orig = np.array(timestamps)
    t_uniform = np.linspace(t_orig[0], t_orig[-1], num_samples)

    rgb_arr = np.array(rgb_weighted_buf)
    r_i = np.interp(t_uniform, t_orig, rgb_arr[:, 0])
    g_i = np.interp(t_uniform, t_orig, rgb_arr[:, 1])
    b_i = np.interp(t_uniform, t_orig, rgb_arr[:, 2])
    rgb_uniform = np.vstack([r_i, g_i, b_i]).T

    sig_dahi = detrend(np.interp(t_uniform, t_orig, np.array(g_dahi_buf)))
    sig_kiri = detrend(np.interp(t_uniform, t_orig, np.array(g_kiri_buf)))
    sig_kanan = detrend(np.interp(t_uniform, t_orig, np.array(g_kanan_buf)))

    # 2. Ekstraksi sinyal POS Jendela Geser + Pre-Emphasis (Turunan Pertama)
    # Turunan pertama (np.gradient) meredam gelombang lambat 40-60 BPM dan mengangkat 80-130 BPM
    pos_raw = compute_ola_pos(rgb_uniform, TARGET_FS)
    pos_preemph = np.gradient(pos_raw)

    try:
        filtered_pos = bandpass_filter(pos_preemph, TARGET_FS, LOW_HZ, HIGH_HZ)
        filt_dahi = bandpass_filter( np.gradient(sig_dahi), TARGET_FS, LOW_HZ, HIGH_HZ)
        filt_kiri = bandpass_filter(np.gradient(sig_kiri), TARGET_FS, LOW_HZ, HIGH_HZ)
        filt_kanan = bandpass_filter(np.gradient(sig_kanan), TARGET_FS, LOW_HZ, HIGH_HZ)
    except ValueError:
        return None

    # 3. Analisis Spektrum Frekuensi (FFT) dengan Koreksi Sub-Harmonik
    windowed = filtered_pos * np.hanning(len(filtered_pos))
    fft_vals = np.abs(np.fft.rfft(windowed))
    fft_freqs = np.fft.rfftfreq(len(filtered_pos), d=1.0 / TARGET_FS)

    valid_idx = np.where((fft_freqs >= LOW_HZ) & (fft_freqs <= HIGH_HZ))[0]
    if len(valid_idx) == 0:
        return None

    valid_freqs = fft_freqs[valid_idx]
    valid_power = (fft_vals[valid_idx] ** 2)

    peak_local_idx = np.argmax(valid_power)
    peak_freq = valid_freqs[peak_local_idx]

    # Cek Sub-Harmonik: Jika puncak terbaca di < 1.15 Hz (< 69 BPM), cek apakah ada puncak
    # harmonik 2x lipatnya (di 90 - 135 BPM) yang cukup kuat
    if peak_freq < 1.15:
        harmonic_band = np.where(
            (valid_freqs >= peak_freq * 1.65) & (valid_freqs <= peak_freq * 2.15)
        )[0]
        if len(harmonic_band) > 0:
            harm_idx = harmonic_band[np.argmax(valid_power[harmonic_band])]
            if valid_power[harm_idx] >= 0.35 * valid_power[peak_local_idx]:
                peak_local_idx = harm_idx
                peak_freq = valid_freqs[peak_local_idx]

    bpm_fft = peak_freq * 60.0

    # 4. Metode Oksimeter: Hitung Jarak Antar-Puncak Gelombang (Inter-Beat Interval / IBI)
    norm_wave = (filtered_pos - np.mean(filtered_pos)) / (np.std(filtered_pos) + 1e-8)
    peaks, _ = find_peaks(norm_wave, distance=int(TARGET_FS * 0.36), prominence=0.45)

    bpm_ibi = bpm_fft
    if len(peaks) >= 4:
        ibi_seconds = np.diff(peaks) / TARGET_FS
        # Ambil IBI yang wajar (0.36s s/d 1.05s atau 57 - 166 BPM)
        valid_ibi = ibi_seconds[(ibi_seconds >= 0.36) & (ibi_seconds <= 1.05)]
        if len(valid_ibi) >= 3:
            bpm_ibi = 60.0 / np.median(valid_ibi)

    # Gabungkan BPM FFT dan BPM Puncak Oksimeter (IBI), lalu haluskan dengan Median Riwayat
    bpm_combined = (0.55 * bpm_fft) + (0.45 * bpm_ibi)
    bpm_history.append(bpm_combined)
    bpm_smooth = float(np.median(bpm_history))

    # 5. Metrik Forensik (SNR & Sinkronisasi ROI)
    peak_band = np.abs(valid_freqs - (bpm_smooth / 60.0)) <= 0.20
    peak_energy = np.sum(valid_power[peak_band])
    total_energy = np.sum(valid_power) + 1e-8
    peak_ratio = peak_energy / total_energy

    corrs = []
    for s_a, s_b in [(filt_dahi, filt_kiri), (filt_dahi, filt_kanan), (filt_kiri, filt_kanan)]:
        if np.std(s_a) > 1e-6 and np.std(s_b) > 1e-6:
            corrs.append(np.corrcoef(s_a, s_b)[0, 1])
    best_sync = max(corrs) if corrs else 0.0

    raw_green_std = np.std(sig_dahi)

    # Cek apakah bacaan sudah sangat stabil seperti oksimeter (deviasi < 1.8 BPM)
    stability_std = np.std(list(bpm_history)[-6:]) if len(bpm_history) >= 6 else 99.0

    return {
        "bpm": bpm_smooth,
        "peak_ratio": peak_ratio,
        "roi_sync": best_sync,
        "signal_std": raw_green_std,
        "stability_std": stability_std,
        "filtered_wave": norm_wave
    }


def draw_signal_graph(canvas, wave_data, x, y, w_box, h_box, locked=False):
    """Menggambar grafik gelombang plethysmograph seperti layar oksimeter."""
    bg_col = (20, 40, 20) if locked else (25, 25, 25)
    border_col = (0, 255, 100) if locked else (110, 110, 110)
    cv2.rectangle(canvas, (x, y), (x + w_box, y + h_box), bg_col, -1)
    cv2.rectangle(canvas, (x, y), (x + w_box, y + h_box), border_col, 2)

    judul_grafik = "Gelombang Pleth Terkunci (Final)" if locked else "Gelombang Denyut Live (Pleth)"
    cv2.putText(canvas, judul_grafik, (x + 12, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

    recent_wave = wave_data[-150:]
    if len(recent_wave) < 2:
        return

    min_v, max_v = np.min(recent_wave), np.max(recent_wave)
    rng = (max_v - min_v) if (max_v - min_v) > 1e-7 else 1.0

    pts = []
    for i, val in enumerate(recent_wave):
        px = x + int((i / (len(recent_wave) - 1)) * (w_box - 24)) + 12
        norm_y = (val - min_v) / rng
        py = y + h_box - 14 - int(norm_y * (h_box - 48))
        pts.append([px, py])

    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 255, 120), 2)


def simpan_ke_csv(csv_path, durasi, res, status_teks):
    """Menyimpan hasil akhir yang sudah terkunci ke file CSV."""
    waktu_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            waktu_str,
            round(durasi, 1),
            round(res["bpm"], 0),
            round(res["peak_ratio"], 3),
            round(res["roi_sync"], 3),
            status_teks
        ])
    print(f"[HASIL AKHIR TERSIMPAN] {waktu_str} | {res['bpm']:.0f} BPM | {status_teks}")


# =====================================================================
# 3. PROGRAM UTAMA (DENGAN FITUR AUTO-HOLD SEPERTI OKSIMETER)
# =====================================================================
CSV_FILE = "data_uji_rppg.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Waktu", "Durasi_Detik", "BPM_rPPG",
            "Peak_Ratio_SNR", "Sinkronisasi_ROI", "Status_Forensik"
        ])

WINDOW_NAME = "UAS Forensik Digital - Mode Oksimeter (Auto-Lock)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

warmup_start = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    h, w, _ = frame.shape
    now = time.time()

    # Perkecil ukuran frame khusus untuk deteksi MediaPipe agar FPS kencang (30 FPS stabil)
    small_bgr = cv2.resize(clean_frame, (480, 270))
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
    face_result = face_detector.detect(mp_image)

    motion_warning = False

    if face_result.face_landmarks:
        landmarks = face_result.face_landmarks[0]

        rgb_dahi, pts_dahi = extract_roi_rgb(clean_frame, landmarks, ROI_DAHI, w, h)
        rgb_kiri, pts_kiri = extract_roi_rgb(clean_frame, landmarks, ROI_PIPI_KIRI, w, h)
        rgb_kanan, pts_kanan = extract_roi_rgb(clean_frame, landmarks, ROI_PIPI_KANAN, w, h)

        poly_color = (0, 255, 255) if is_locked else (0, 255, 0)
        for pts, label in [
            (pts_dahi, "Dahi (50%)"),
            (pts_kiri, "Pipi Kiri (25%)"),
            (pts_kanan, "Pipi Kanan (25%)")
        ]:
            cv2.polylines(frame, [pts], True, poly_color, 2)
            cx, cy = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, label, (cx - 35, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if warmup_start is None:
            warmup_start = now

        # Hanya kumpulkan data jika hasil BELUM dikunci (seperti oksimeter sedang mengukur)
        if not is_locked and (now - warmup_start >= WARMUP_SECONDS):
            nx, ny = landmarks[1].x * w, landmarks[1].y * h
            nose_pos_buf.append((nx, ny))

            rgb_weighted = (0.50 * rgb_dahi) + (0.25 * rgb_kiri) + (0.25 * rgb_kanan)

            timestamps.append(now)
            rgb_weighted_buf.append(rgb_weighted)
            g_dahi_buf.append(rgb_dahi[1])
            g_kiri_buf.append(rgb_kiri[1])
            g_kanan_buf.append(rgb_kanan[1])

            if len(nose_pos_buf) > 15:
                recent_nose = np.array(list(nose_pos_buf)[-15:])
                movement = np.max(np.std(recent_nose, axis=0))
                if movement > (w * 0.015):
                    motion_warning = True
    else:
        if not is_locked:
            warmup_start = None
            reset_measurement()

    display_canvas = cv2.resize(frame, (1280, 720))
    disp_h, disp_w, _ = display_canvas.shape

    # Hitung hasil jika belum terkunci
    if not is_locked:
        elapsed = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        current_res = analyze_rppg_oximeter()

        # Syarat Penguncian Otomatis (Auto-Lock seperti Oksimeter):
        # 1. Sudah mengukur minimal 12 detik DAN sinyal sangat stabil (fluktuasi < 1.5 BPM), ATAU
        # 2. Sudah mencapai batas waktu ukur 18 detik
        if current_res is not None:
            if (elapsed >= 12.0 and current_res["stability_std"] <= 1.5) or (elapsed >= LOCK_SECONDS):
                is_locked = True
                locked_result = current_res
                locked_duration = elapsed

                pr = current_res["peak_ratio"]
                sync = current_res["roi_sync"]
                std_val = current_res["signal_std"]
                bpm_val = current_res["bpm"]

                is_live = (
                    (pr >= 0.11)
                    and (0.05 <= std_val <= 12.0)
                    and (sync >= -0.05)
                    and (55.0 <= bpm_val <= 175.0)
                )

                if is_live:
                    locked_status = "WAJAH ASLI / LIVE (Denyut Terverifikasi)"
                    locked_color = (0, 255, 100)
                else:
                    locked_status = "INDIKASI SPOOF / FOTO / LAYAR"
                    locked_color = (0, 80, 255)

                simpan_ke_csv(CSV_FILE, locked_duration, locked_result, locked_status)

    # =================================================================
    # TAMPILAN PANEL ATAS (BERBEDA SAAT MENGUKUR VS SAAT HASIL TERKUNCI)
    # =================================================================
    cv2.rectangle(display_canvas, (0, 0), (disp_w, 135), (18, 18, 18), -1)

    if is_locked and locked_result is not None:
        # TAMPILAN SAAT HASIL SUDAH DIKUNCI (BERHENTI SEPERTI OKSIMETER)
        cv2.rectangle(display_canvas, (0, 0), (disp_w, 135), (15, 45, 20), -1)
        cv2.putText(
            display_canvas,
            f"PENGUKURAN SELESAI & TERKUNCI ({locked_duration:.1f} dtk)  |  Tekan [SPASI] atau [R] untuk Ukur Ulang  |  [Q] Keluar",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 180), 2
        )
        cv2.putText(
            display_canvas,
            f"HASIL AKHIR: {locked_result['bpm']:.0f} BPM",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 255), 3
        )
        cv2.putText(
            display_canvas,
            f"Forensik: {locked_status}",
            (490, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.78, locked_color, 2
        )
        cv2.putText(
            display_canvas,
            f"Kejelasan Puncak (SNR): {locked_result['peak_ratio']*100:.1f}%   |   Sinkronisasi ROI: {locked_result['roi_sync']:.2f}   |   Otomatis Tersimpan ke CSV",
            (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1
        )
        draw_signal_graph(display_canvas, locked_result["filtered_wave"], disp_w - 430, disp_h - 155, 410, 135, locked=True)

    else:
        # TAMPILAN SAAT PROSES MENGUKUR BERJALAN
        elapsed = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        progress_pct = min(100, int((elapsed / LOCK_SECONDS) * 100))

        cv2.putText(
            display_canvas,
            f"Proses Mengukur Oksimeter: {elapsed:.1f} / {LOCK_SECONDS:.0f} dtk ({progress_pct}%)  |  [R / SPASI] Ulangi  |  [Q] Keluar",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1
        )

        if warmup_start is not None and (now - warmup_start < WARMUP_SECONDS):
            sisa_warmup = WARMUP_SECONDS - (now - warmup_start)
            cv2.putText(
                display_canvas,
                f"Kalibrasi Cahaya & Wajah ({sisa_warmup:.1f} dtk)... Mohon diam sebentar.",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 220, 255), 2
            )
        elif current_res is not None:
            cv2.putText(
                display_canvas,
                f"Mengukur Detak: ~{current_res['bpm']:.0f} BPM (Menstabilkan bacaan...)",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2
            )
            cv2.putText(
                display_canvas,
                f"SNR Saat Ini: {current_res['peak_ratio']*100:.1f}%   |   Sinkronisasi: {current_res['roi_sync']:.2f}   |   Hasil akan dikunci otomatis saat stabil",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
            )
            draw_signal_graph(display_canvas, current_res["filtered_wave"], disp_w - 430, disp_h - 155, 410, 135, locked=False)
        else:
            cv2.putText(
                display_canvas,
                f"Mendeteksi denyut nadi ({elapsed:.1f} dtk / min {MIN_MEASURE_SEC:.0f} dtk)...",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 200, 255), 2
            )
            cv2.putText(
                display_canvas,
                "Posisikan wajah menghadap kamera dan tahan gerakan selama beberapa detik.",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1
            )

    if motion_warning and not is_locked:
        cv2.putText(display_canvas, "PERINGATAN: Gerakan kepala terdeteksi! Tahan posisi wajah.",
                    (20, disp_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, display_canvas)

    k = cv2.waitKey(1) & 0xff
    if k == ord('r') or k == ord(' '):
        warmup_start = time.time()
        reset_measurement()
        print("Mengulang pengukuran baru...")
    elif k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()