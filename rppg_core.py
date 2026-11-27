"""
rppg_core.py
=====================================================================
MODUL INTI PERHITUNGAN rPPG (dipakai oleh rppgforensik_v2.py DAN
analisis_validasi.py, sehingga perhitungan di kamera == perhitungan
ulang di luar kamera / offline).

Isi modul:
  1. resample_uniform()   -> interpolasi sinyal ke 30 Hz seragam
  2. pos_wang2017()       -> algoritma POS (Wang et al., 2017) + langkah perantara
  3. chrom_dehaan2013()   -> algoritma CHROM (de Haan & Jeanne, 2013) pembanding
  4. green_verkruysse()   -> metode kanal hijau (Verkruysse et al., 2008) pembanding
  5. bandpass()           -> filter Butterworth band-pass 0.75 - 3.0 Hz
  6. spectrum_bpm()       -> estimasi BPM dari spektrum Welch + interpolasi parabola
  7. snr_dehaan()         -> SNR (dB) definisi de Haan & Jeanne (2013)
  8. ibi_bpm()            -> BPM dari jarak antar-puncak (Inter-Beat Interval)
  9. ita_from_bgr()       -> warna kulit: Individual Typology Angle (Chardon, 1991)
 10. analyze_window()     -> pipeline lengkap, mengembalikan SEMUA tahap perantara
 11. plot_pipeline()      -> gambar 6 panel: perubahan sinyal dari awal s.d. akhir
=====================================================================
"""
import numpy as np
from scipy.signal import butter, filtfilt, detrend, welch, find_peaks

# --------------------------------------------------------------------
# PARAMETER GLOBAL (semua angka di sini punya dasar literatur)
# --------------------------------------------------------------------
TARGET_FS = 30.0   # Hz, frekuensi sampling seragam setelah interpolasi
LOW_HZ = 0.75      # 45 BPM  (batas bawah detak jantung istirahat dewasa)
HIGH_HZ = 3.00     # 180 BPM (batas atas)
POS_WIN_SEC = 1.6  # panjang jendela POS (Wang et al., 2017: l = 1.6 detik @ 20 fps -> 32 frame)
NFFT = 8192        # zero-padding agar resolusi tampilan spektrum halus (~0.22 BPM / bin)

# Matriks proyeksi POS (Wang et al., 2017):  S = P . Cn
P_POS = np.array([[0.0, 1.0, -1.0],
                  [-2.0, 1.0, 1.0]])


# =====================================================================
# 1. INTERPOLASI KE SAMPLING SERAGAM
# =====================================================================
def resample_uniform(t, x, fs=TARGET_FS):
    """
    Kamera tidak menghasilkan frame dengan jarak waktu konstan
    (FPS naik-turun). FFT mensyaratkan jarak sampel seragam, maka
    sinyal diinterpolasi linear ke grid waktu seragam 1/fs detik.
    t : array waktu (detik), x : array (N,) atau (N, k)
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    xu = np.column_stack([np.interp(tu, t, x[:, k]) for k in range(x.shape[1])])
    return tu, xu


# =====================================================================
# 2. FILTER BAND-PASS
# =====================================================================
def bandpass(x, fs=TARGET_FS, low=LOW_HZ, high=HIGH_HZ, order=3):
    """Butterworth orde-3, zero-phase (filtfilt) -> tidak menggeser fase/puncak."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


# =====================================================================
# 3. ALGORITMA POS (Plane-Orthogonal-to-Skin), Wang et al. 2017
# =====================================================================
def pos_wang2017(rgb, fs=TARGET_FS, win_sec=POS_WIN_SEC, return_steps=False):
    """
    Implementasi Algoritma 1 pada Wang et al. (2017), IEEE TBME 64(7).

    Untuk setiap jendela geser sepanjang l sampel:
      (a) Normalisasi temporal : Cn = C / mean(C)          -> menghapus warna kulit dasar (DC)
      (b) Proyeksi             : S1 = Gn - Bn
                                 S2 = -2Rn + Gn + Bn
      (c) Tuning alfa          : h  = S1 + (std(S1)/std(S2)) * S2
      (d) Overlap-add          : H[n-l : n] += h - mean(h)

    Normalisasi (a) membuat POS tidak tergantung pada warna kulit absolut,
    karena yang dianalisis adalah PERUBAHAN RELATIF warna (AC/DC).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    l = max(8, int(round(win_sec * fs)))
    H = np.zeros(n)
    alphas = np.zeros(n)
    s1_full = np.zeros(n)
    s2_full = np.zeros(n)
    cnt = np.zeros(n)

    for m in range(l, n + 1):
        s = m - l
        C = rgb[s:m]                              # (l, 3)
        mu = C.mean(axis=0)
        if np.any(mu <= 1e-6):
            continue
        Cn = C / mu                               # (a)
        S = Cn @ P_POS.T                          # (b) -> (l, 2)
        s1, s2 = S[:, 0], S[:, 1]
        alpha = np.std(s1) / (np.std(s2) + 1e-12)  # (c)
        h = s1 + alpha * s2
        H[s:m] += h - h.mean()                    # (d)
        # simpan nilai perantara (untuk ditampilkan/dilaporkan)
        s1_full[s:m] += s1 - s1.mean()
        s2_full[s:m] += s2 - s2.mean()
        alphas[m - 1] = alpha
        cnt[s:m] += 1

    if not return_steps:
        return H
    cnt[cnt == 0] = 1
    return H, {"S1": s1_full / cnt, "S2": s2_full / cnt, "alpha": alphas}


# =====================================================================
# 4. METODE PEMBANDING: CHROM dan GREEN
# =====================================================================
def chrom_dehaan2013(rgb, fs=TARGET_FS):
    """
    CHROM (de Haan & Jeanne, 2013):
      Xs = 3Rn - 2Gn ; Ys = 1.5Rn + Gn - 1.5Bn
      S  = Xf - (std(Xf)/std(Yf)) * Yf   (Xf, Yf = hasil band-pass)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    Cn = rgb / (rgb.mean(axis=0) + 1e-12)
    Xs = 3.0 * Cn[:, 0] - 2.0 * Cn[:, 1]
    Ys = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
    Xf, Yf = bandpass(Xs, fs), bandpass(Ys, fs)
    return Xf - (np.std(Xf) / (np.std(Yf) + 1e-12)) * Yf


def green_verkruysse(rgb, fs=TARGET_FS):
    """GREEN (Verkruysse et al., 2008): kanal hijau ternormalisasi, di-detrend."""
    g = np.asarray(rgb, dtype=np.float64)[:, 1]
    return detrend(g / (g.mean() + 1e-12) - 1.0)


# =====================================================================
# 5. ESTIMASI BPM DARI SPEKTRUM (Welch PSD)
# =====================================================================
def spectrum_bpm(x, fs=TARGET_FS, low=LOW_HZ, high=HIGH_HZ):
    """
    Power Spectral Density metode Welch (rata-rata beberapa segmen
    berjendela Hann) -> lebih stabil daripada satu FFT tunggal.
    Puncak dicari pada pita 0.75-3.0 Hz, lalu dihaluskan dengan
    interpolasi parabola 3 titik agar presisi di bawah 1 bin.

    BPM = f_puncak (Hz) x 60
    """
    n = len(x)
    nperseg = min(n, int(8 * fs))                 # segmen 8 detik
    noverlap = int(nperseg * 0.75)
    f, pxx = welch(x, fs=fs, window="hann", nperseg=nperseg,
                   noverlap=noverlap, nfft=max(NFFT, nperseg), detrend="constant")
    band = (f >= low) & (f <= high)
    fb, pb = f[band], pxx[band]
    k = int(np.argmax(pb))
    f_peak = fb[k]
    if 0 < k < len(pb) - 1:                       # interpolasi parabola
        y0, y1, y2 = np.log(pb[k - 1] + 1e-20), np.log(pb[k] + 1e-20), np.log(pb[k + 1] + 1e-20)
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            f_peak = fb[k] + delta * (fb[1] - fb[0])
    return float(f_peak * 60.0), f, pxx


def snr_dehaan(f, pxx, f_hr, low=LOW_HZ, high=HIGH_HZ):
    """
    SNR (de Haan & Jeanne, 2013):
      energi di [f_hr +- 0.1 Hz] + [2 f_hr +- 0.2 Hz]
      dibagi energi sisanya pada pita 0.75-4 Hz, dalam desibel.
    Wajah asli -> SNR tinggi (puncak jelas). Foto -> SNR rendah.
    """
    band = (f >= low) & (f <= 4.0)
    fb, pb = f[band], pxx[band]
    sig = (np.abs(fb - f_hr) <= 0.1) | (np.abs(fb - 2 * f_hr) <= 0.2)
    ps, pn = pb[sig].sum(), pb[~sig].sum()
    return float(10.0 * np.log10((ps + 1e-20) / (pn + 1e-20)))


def ibi_bpm(x, fs=TARGET_FS, bpm_hint=None):
    """
    Metode cara kerja oksimeter: cari puncak tiap denyut, ukur jarak
    antar-puncak (IBI, detik), BPM = 60 / median(IBI).
    Jarak minimum antar-puncak disesuaikan dengan BPM spektral (hint)
    agar puncak ganda (dicrotic notch) tidak terhitung.
    """
    z = (x - np.mean(x)) / (np.std(x) + 1e-12)
    if bpm_hint:
        min_dist = int(fs * 0.6 * 60.0 / bpm_hint)
    else:
        min_dist = int(fs * 60.0 / 180.0)
    peaks, _ = find_peaks(z, distance=max(1, min_dist), prominence=0.3)
    if len(peaks) < 4:
        return None, peaks, np.array([])
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi >= 60.0 / 180.0) & (ibi <= 60.0 / 45.0)]
    if len(ibi) < 3:
        return None, peaks, ibi
    return float(60.0 / np.median(ibi)), peaks, ibi


# =====================================================================
# 6. WARNA KULIT: INDIVIDUAL TYPOLOGY ANGLE (ITA)
# =====================================================================
ITA_KATEGORI = [  # (batas bawah, nama)  Chardon et al. 1991 / Del Bino 2013
    (55.0, "Sangat Terang (Very Light)"),
    (41.0, "Terang (Light)"),
    (28.0, "Sedang (Intermediate)"),
    (10.0, "Sawo Matang (Tan)"),
    (-30.0, "Cokelat (Brown)"),
    (-999.0, "Gelap (Dark)"),
]


def ita_from_bgr(pixels_bgr):
    """
    pixels_bgr : array (N,3) uint8 piksel kulit.
    Konversi ke CIELAB (OpenCV, float32 0-1 -> L 0..100, b* asli),
    ITA = arctan((L* - 50) / b*) x 180/pi   (Chardon et al., 1991)
    Mengembalikan (ITA_derajat, L*, a*, b*, kategori).
    CATATAN: webcam tidak terkalibrasi warna -> ITA bersifat relatif
    (pembanding antar-subjek pada pencahayaan yang SAMA).
    """
    import cv2
    px = np.asarray(pixels_bgr, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    L, a, b = np.median(lab[:, 0]), np.median(lab[:, 1]), np.median(lab[:, 2])
    ita = float(np.degrees(np.arctan2(L - 50.0, b)))  # arctan2 aman bila b* ~ 0
    kategori = next(nama for batas, nama in ITA_KATEGORI if ita > batas)
    return ita, float(L), float(a), float(b), kategori


# =====================================================================
# 7. PIPELINE LENGKAP SATU JENDELA ANALISIS
# =====================================================================
def analyze_window(t, rgb_weighted, rgb_rois=None, fs=TARGET_FS):
    """
    t             : (N,) waktu frame (detik)
    rgb_weighted  : (N,3) rata-rata RGB gabungan ROI berbobot
    rgb_rois      : dict {"dahi":(N,3), "pipi_kiri":(N,3), "pipi_kanan":(N,3)} (opsional)

    Mengembalikan dict berisi SEMUA tahap perantara, sehingga setiap
    perubahan sinyal dari awal sampai akhir bisa ditampilkan & disimpan.
    """
    tu, rgb_u = resample_uniform(t, rgb_weighted, fs)            # TAHAP 1
    if len(tu) < int(6 * fs):
        return None
    mu = rgb_u.mean(axis=0)
    rgb_norm = rgb_u / mu                                         # TAHAP 2 (Cn)
    pos_raw, steps = pos_wang2017(rgb_u, fs, return_steps=True)   # TAHAP 3
    pos_f = bandpass(detrend(pos_raw), fs)                        # TAHAP 4
    bpm_spec, f, pxx = spectrum_bpm(pos_f, fs)                    # TAHAP 5
    snr = snr_dehaan(f, pxx, bpm_spec / 60.0)
    bpm_ibi, peaks, ibi = ibi_bpm(pos_f, fs, bpm_hint=bpm_spec)   # TAHAP 6

    # Metode pembanding (untuk tabel perbandingan di laporan)
    bpm_chrom, _, _ = spectrum_bpm(chrom_dehaan2013(rgb_u, fs), fs)
    bpm_green, _, _ = spectrum_bpm(bandpass(green_verkruysse(rgb_u, fs), fs), fs)

    # Metrik forensik antar-ROI (sinkronisasi denyut dahi vs pipi)
    roi_bpm, roi_corr = {}, None
    if rgb_rois:
        roi_sig = {}
        for name, arr in rgb_rois.items():
            _, arr_u = resample_uniform(t, arr, fs)
            sig = bandpass(detrend(pos_wang2017(arr_u, fs)), fs)
            roi_sig[name] = sig
            roi_bpm[name] = spectrum_bpm(sig, fs)[0]
        names = list(roi_sig)
        cs = [np.corrcoef(roi_sig[a], roi_sig[b])[0, 1]
              for i, a in enumerate(names) for b in names[i + 1:]]
        roi_corr = float(np.mean(cs)) if cs else None

    bpm_vals = list(roi_bpm.values())
    roi_spread = float(np.max(bpm_vals) - np.min(bpm_vals)) if bpm_vals else None

    return {
        "t_uniform": tu, "rgb_uniform": rgb_u, "rgb_mean": mu, "rgb_norm": rgb_norm,
        "S1": steps["S1"], "S2": steps["S2"], "alpha": steps["alpha"],
        "pos_raw": pos_raw, "pos_filtered": pos_f,
        "freqs": f, "psd": pxx,
        "bpm": bpm_spec, "bpm_ibi": bpm_ibi, "peaks": peaks, "ibi": ibi,
        "bpm_chrom": bpm_chrom, "bpm_green": bpm_green,
        "snr_db": snr, "roi_bpm": roi_bpm, "roi_corr": roi_corr,
        "roi_bpm_spread": roi_spread,
        "fs_eff": (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0,
    }


# =====================================================================
# 8. KLASIFIKASI LIVENESS (aturan berbasis ambang, ambang bisa dikalibrasi)
# =====================================================================
DEFAULT_THRESH = {"snr_db": -3.0, "roi_corr": 0.30, "roi_spread": 8.0}


def classify_liveness(res, thr=DEFAULT_THRESH):
    """
    Wajah asli (live) bila:
      1) SNR >= ambang (puncak denyut jelas),
      2) korelasi sinyal antar-ROI >= ambang (aliran darah sinkron),
      3) selisih BPM antar-ROI <= ambang (semua bagian wajah berdenyut sama).
    Ambang awal = perkiraan; WAJIB dikalibrasi ulang dari data kelompok
    dengan analisis_validasi.py (bagian ROC / Youden J).
    """
    ok_snr = res["snr_db"] >= thr["snr_db"]
    ok_corr = (res["roi_corr"] is not None) and (res["roi_corr"] >= thr["roi_corr"])
    ok_spread = (res["roi_bpm_spread"] is not None) and (res["roi_bpm_spread"] <= thr["roi_spread"])
    alasan = {"SNR": ok_snr, "Korelasi ROI": ok_corr, "Konsistensi BPM ROI": ok_spread}
    return (ok_snr and ok_corr and ok_spread), alasan


# =====================================================================
# 9. GAMBAR PIPELINE (6 PANEL) -> BUKTI "DARI AWAL SAMPAI AKHIR"
# =====================================================================
def plot_pipeline(res, out_png, judul="", bpm_ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tu = res["t_uniform"] - res["t_uniform"][0]
    fig, ax = plt.subplots(6, 1, figsize=(11, 15))
    fig.suptitle(judul or "Pipeline rPPG (POS) - Tahap demi Tahap", fontsize=13, weight="bold")

    c = res["rgb_uniform"]
    for k, (col, nm) in enumerate(zip(["tab:red", "tab:green", "tab:blue"], "RGB")):
        ax[0].plot(tu, c[:, k], color=col, lw=1, label=f"{nm} (rata2={res['rgb_mean'][k]:.1f})")
    ax[0].set_title("Tahap 1 - Rata-rata RGB ROI berbobot (50% dahi, 25% pipi kiri, 25% pipi kanan), interpolasi 30 Hz")
    ax[0].set_ylabel("Intensitas (0-255)"); ax[0].legend(fontsize=8, loc="upper right")

    for k, (col, nm) in enumerate(zip(["tab:red", "tab:green", "tab:blue"], "RGB")):
        ax[1].plot(tu, res["rgb_norm"][:, k], color=col, lw=1, label=f"{nm}n = {nm}/mean({nm})")
    ax[1].set_title("Tahap 2 - Normalisasi temporal Cn = C / mean(C)  (menghapus pengaruh warna kulit dasar)")
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].plot(tu, res["S1"], lw=1, label="S1 = Gn - Bn")
    ax[2].plot(tu, res["S2"], lw=1, label="S2 = -2Rn + Gn + Bn")
    ax[2].set_title("Tahap 3a - Proyeksi ke bidang ortogonal kulit (POS)")
    ax[2].legend(fontsize=8, loc="upper right")

    ax[3].plot(tu, res["pos_raw"], color="purple", lw=1, label="H = S1 + (std S1 / std S2) * S2  (overlap-add 1.6 s)")
    ax[3].set_title("Tahap 3b - Sinyal pulsa POS mentah")
    ax[3].legend(fontsize=8, loc="upper right")

    pf = res["pos_filtered"]
    ax[4].plot(tu, pf, color="black", lw=1, label="Band-pass Butterworth 0.75-3.0 Hz")
    if len(res["peaks"]):
        ax[4].plot(tu[res["peaks"]], pf[res["peaks"]], "rv", ms=6, label=f"Puncak denyut (n={len(res['peaks'])})")
    ibi_txt = f"{res['bpm_ibi']:.1f} BPM" if res["bpm_ibi"] else "-"
    ax[4].set_title(f"Tahap 4 - Sinyal terfilter + deteksi puncak (IBI -> {ibi_txt})")
    ax[4].set_xlabel("Waktu (detik)"); ax[4].legend(fontsize=8, loc="upper right")

    f, p = res["freqs"], res["psd"]
    m = (f >= 0.5) & (f <= 4.0)
    ax[5].plot(f[m] * 60, p[m] / p[m].max(), color="tab:orange")
    ax[5].axvspan(45, 180, color="green", alpha=0.06, label="Pita valid 45-180 BPM")
    ax[5].axvline(res["bpm"], color="red", ls="--", label=f"rPPG = {res['bpm']:.1f} BPM")
    if bpm_ref:
        ax[5].axvline(bpm_ref, color="blue", ls=":", label=f"Oksimeter = {bpm_ref:.0f} BPM")
    ax[5].set_title(f"Tahap 5 - Spektrum daya Welch  |  SNR = {res['snr_db']:.2f} dB")
    ax[5].set_xlabel("Frekuensi (BPM = Hz x 60)"); ax[5].set_ylabel("Daya ternormalisasi")
    ax[5].legend(fontsize=8, loc="upper right")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
