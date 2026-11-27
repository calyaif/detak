"""
analisis_validasi.py  —  Bukti data untuk laporan UAS
=====================================================================
Dua mode:

1) REKAP STATISTIK SEMUA SESI (default)
   python analisis_validasi.py
   -> membaca data_uji_rppg_v2.csv, menghasilkan folder laporan_validasi/ berisi:
      - tabel_per_kondisi.csv     : MAE, RMSE, MAPE, bias, LoA, r per kondisi
      - tabel_per_warna_kulit.csv : metrik yang sama per kategori kulit ITA
      - tabel_per_metode.csv      : POS vs CHROM vs GREEN vs IBI
      - scatter_rppg_vs_oksimeter.png, bland_altman.png
      - error_per_warna_kulit.png
      - liveness_snr.png + metrik APCER/BPCER/akurasi (ISO/IEC 30107-3)
      - ringkasan.txt

2) REKONSTRUKSI SATU SESI (bukti perhitungan "di luar kamera")
   python analisis_validasi.py --rekonstruksi hasil_sesi/20261126_101500_S01_live
   -> menghitung ULANG dari raw_signal.csv, mencetak angka tiap tahap
      (contoh hitung manual), lalu membuat rekonstruksi_pipeline.png.
=====================================================================
"""
import os
import csv
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rppg_core as rc

BASE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# UTIL
# =====================================================================
def baca_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def metrik(est, ref):
    """Metrik akurasi standar (dipakai di hampir semua paper rPPG)."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    m = ~(np.isnan(est) | np.isnan(ref))
    est, ref = est[m], ref[m]
    n = len(est)
    if n == 0:
        return None
    e = est - ref
    bias, sd = float(np.mean(e)), float(np.std(e, ddof=1)) if n > 1 else 0.0
    r = float(np.corrcoef(est, ref)[0, 1]) if n > 2 and np.std(est) > 0 and np.std(ref) > 0 else np.nan
    return {
        "N": n,
        "MAE": float(np.mean(np.abs(e))),                 # rata-rata |rPPG - oksimeter|
        "RMSE": float(np.sqrt(np.mean(e ** 2))),           # menghukum error besar
        "MAPE_%": float(np.mean(np.abs(e) / ref) * 100),   # error relatif
        "Bias": bias,                                       # Bland-Altman: rata-rata selisih
        "SD": sd,
        "LoA_bawah": bias - 1.96 * sd,                      # batas kesepakatan 95%
        "LoA_atas": bias + 1.96 * sd,
        "Pearson_r": r,
        "Persen_err<=5BPM": float(np.mean(np.abs(e) <= 5) * 100),
    }


def tulis_tabel(path, baris):
    if not baris:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        wr.writeheader()
        for b in baris:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})


# =====================================================================
# MODE 1: REKAP STATISTIK
# =====================================================================
def rekap(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = baca_csv(csv_path)
    log = []

    def p(s=""):
        print(s)
        log.append(s)

    p(f"Total sesi: {len(rows)}  (sumber: {csv_path})")

    # ------------ A. AKURASI BPM (hanya wajah asli & ada oksimeter) ------------
    live = [r for r in rows if r["Label_Asli"] == "live" and fnum(r["BPM_Oksimeter"]) > 0]
    p(f"Sesi wajah asli dengan referensi oksimeter: {len(live)}")
    if live:
        est = np.array([fnum(r["BPM_rPPG_POS"]) for r in live])
        ref = np.array([fnum(r["BPM_Oksimeter"]) for r in live])

        tab_k = []
        for k in sorted(set(r["Kondisi"] for r in live)):
            sub = [r for r in live if r["Kondisi"] == k]
            m = metrik([fnum(r["BPM_rPPG_POS"]) for r in sub], [fnum(r["BPM_Oksimeter"]) for r in sub])
            tab_k.append({"Kondisi": k, **m})
        m_all = metrik(est, ref)
        tab_k.append({"Kondisi": "SEMUA", **m_all})
        tulis_tabel(os.path.join(out_dir, "tabel_per_kondisi.csv"), tab_k)

        tab_s = []
        for k in sorted(set(r["Kategori_Kulit_ITA"] for r in live)):
            sub = [r for r in live if r["Kategori_Kulit_ITA"] == k]
            m = metrik([fnum(r["BPM_rPPG_POS"]) for r in sub], [fnum(r["BPM_Oksimeter"]) for r in sub])
            ita = np.nanmean([fnum(r["ITA_derajat"]) for r in sub])
            snr = np.nanmean([fnum(r["SNR_dB"]) for r in sub])
            tab_s.append({"Kategori_Kulit": k, "ITA_rata2": ita, "SNR_rata2_dB": snr, **m})
        tulis_tabel(os.path.join(out_dir, "tabel_per_warna_kulit.csv"), tab_s)

        tab_m = []
        for kol, nama in [("BPM_rPPG_POS", "POS (Wang 2017)"), ("BPM_CHROM", "CHROM (de Haan 2013)"),
                          ("BPM_GREEN", "GREEN (Verkruysse 2008)"), ("BPM_IBI", "IBI puncak (POS)")]:
            m = metrik([fnum(r[kol]) for r in live], ref)
            if m:
                tab_m.append({"Metode": nama, **m})
        tulis_tabel(os.path.join(out_dir, "tabel_per_metode.csv"), tab_m)

        p("\n=== AKURASI BPM rPPG (POS) vs OKSIMETER ===")
        for t in tab_k:
            p(f"  {t['Kondisi']:8s} N={t['N']:3d}  MAE={t['MAE']:.2f}  RMSE={t['RMSE']:.2f}  "
              f"MAPE={t['MAPE_%']:.1f}%  bias={t['Bias']:+.2f}  LoA=[{t['LoA_bawah']:.1f}, {t['LoA_atas']:.1f}]  r={t['Pearson_r']:.3f}")
        p("\n=== PER WARNA KULIT (ITA) ===")
        for t in tab_s:
            p(f"  {t['Kategori_Kulit']:28s} N={t['N']:3d}  ITA={t['ITA_rata2']:6.1f}  SNR={t['SNR_rata2_dB']:6.2f} dB  MAE={t['MAE']:.2f}")
        p("\n=== PERBANDINGAN METODE ===")
        for t in tab_m:
            p(f"  {t['Metode']:26s} N={t['N']:3d}  MAE={t['MAE']:.2f}  RMSE={t['RMSE']:.2f}  r={t['Pearson_r']:.3f}")

        # --- grafik scatter + Bland-Altman
        kat = [r["Kondisi"] for r in live]
        warna = {"live": "tab:green", "makeup": "tab:purple"}
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        for k in set(kat):
            idx = [i for i, x in enumerate(kat) if x == k]
            ax[0].scatter(ref[idx], est[idx], label=k, c=warna.get(k, "gray"), s=40, alpha=0.8)
        lo, hi = min(ref.min(), est.min()) - 5, max(ref.max(), est.max()) + 5
        ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (sempurna)")
        ax[0].fill_between([lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5], color="gray", alpha=0.12, label="±5 BPM")
        ax[0].set_xlabel("BPM Oksimeter (referensi)"); ax[0].set_ylabel("BPM rPPG")
        ax[0].set_title(f"rPPG vs Oksimeter  (r = {m_all['Pearson_r']:.3f}, MAE = {m_all['MAE']:.2f})")
        ax[0].legend(); ax[0].grid(alpha=0.3)

        mean_ = (est + ref) / 2
        diff = est - ref
        for k in set(kat):
            idx = [i for i, x in enumerate(kat) if x == k]
            ax[1].scatter(mean_[idx], diff[idx], c=warna.get(k, "gray"), s=40, alpha=0.8, label=k)
        ax[1].axhline(m_all["Bias"], color="red", label=f"Bias {m_all['Bias']:+.2f}")
        ax[1].axhline(m_all["LoA_atas"], color="red", ls="--", label=f"+1.96SD {m_all['LoA_atas']:.1f}")
        ax[1].axhline(m_all["LoA_bawah"], color="red", ls="--", label=f"-1.96SD {m_all['LoA_bawah']:.1f}")
        ax[1].set_xlabel("Rata-rata (rPPG + Oksimeter)/2"); ax[1].set_ylabel("Selisih rPPG - Oksimeter")
        ax[1].set_title("Bland-Altman Plot"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "scatter_dan_bland_altman.png"), dpi=120); plt.close(fig)

        # --- error per warna kulit
        if tab_s:
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = [t["Kategori_Kulit"].split(" (")[0] for t in tab_s]
            ax.bar(labels, [t["MAE"] for t in tab_s], color="tab:orange")
            for i, t in enumerate(tab_s):
                ax.text(i, t["MAE"], f"MAE {t['MAE']:.1f}\nN={t['N']}\nSNR {t['SNR_rata2_dB']:.1f} dB", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("MAE (BPM)"); ax.set_title("Error rPPG per Kategori Warna Kulit (ITA)")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, "error_per_warna_kulit.png"), dpi=120); plt.close(fig)

    # ------------ B. LIVENESS DETECTION ------------
    y = np.array([1 if r["Label_Asli"] == "live" else 0 for r in rows])
    yhat = np.array([1 if r["Prediksi_Liveness"] == "live" else 0 for r in rows])
    if len(set(y)) == 2:
        tp = int(np.sum((y == 1) & (yhat == 1))); fn = int(np.sum((y == 1) & (yhat == 0)))
        tn = int(np.sum((y == 0) & (yhat == 0))); fp = int(np.sum((y == 0) & (yhat == 1)))
        apcer = fp / max(1, fp + tn) * 100   # serangan yang lolos dianggap asli
        bpcer = fn / max(1, fn + tp) * 100   # wajah asli yang ditolak
        acc = (tp + tn) / len(y) * 100
        p("\n=== LIVENESS (ISO/IEC 30107-3) ===")
        p(f"  Confusion: TP={tp} FN={fn} | FP={fp} TN={tn}")
        p(f"  Akurasi={acc:.1f}%  APCER={apcer:.1f}%  BPCER={bpcer:.1f}%  ACER={(apcer + bpcer) / 2:.1f}%")
        for k in sorted(set(r["Kondisi"] for r in rows)):
            sub = [r for r in rows if r["Kondisi"] == k]
            benar = sum(1 for r in sub if r["Prediksi_Liveness"] == r["Label_Asli"])
            p(f"    {k:8s}: {benar}/{len(sub)} benar")

        # kalibrasi ambang SNR terbaik (Youden J = TPR - FPR)
        snr = np.array([fnum(r["SNR_dB"]) for r in rows])
        best = None
        for th in np.unique(snr[~np.isnan(snr)]):
            pr = snr >= th
            tpr = np.mean(pr[y == 1]); fpr = np.mean(pr[y == 0])
            if best is None or tpr - fpr > best[1]:
                best = (th, tpr - fpr, tpr, fpr)
        if best:
            p(f"  Ambang SNR optimal (Youden J): {best[0]:.2f} dB  (TPR={best[2] * 100:.0f}%, FPR={best[3] * 100:.0f}%)")
            p("  -> salin nilai ini ke DEFAULT_THRESH['snr_db'] di rppg_core.py")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        kondisi = sorted(set(r["Kondisi"] for r in rows))
        data_snr = [[fnum(r["SNR_dB"]) for r in rows if r["Kondisi"] == k] for k in kondisi]
        data_spr = [[fnum(r["Selisih_BPM_ROI"]) for r in rows if r["Kondisi"] == k] for k in kondisi]
        ax[0].boxplot(data_snr); ax[0].set_xticks(range(1, len(kondisi) + 1), kondisi)
        if best:
            ax[0].axhline(best[0], color="red", ls="--", label=f"ambang {best[0]:.2f} dB")
            ax[0].legend()
        ax[0].set_title("SNR sinyal denyut per kondisi"); ax[0].set_ylabel("SNR (dB)"); ax[0].grid(alpha=0.3)
        ax[1].boxplot(data_spr); ax[1].set_xticks(range(1, len(kondisi) + 1), kondisi)
        ax[1].set_title("Selisih BPM antar-ROI (dahi/pipi)"); ax[1].set_ylabel("BPM"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "liveness_snr.png"), dpi=120); plt.close(fig)
    else:
        p("\n(Liveness belum bisa dievaluasi: butuh data live DAN spoof)")

    with open(os.path.join(out_dir, "ringkasan.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"\nSemua tabel & grafik tersimpan di: {out_dir}")


# =====================================================================
# MODE 2: REKONSTRUKSI SATU SESI (hitung ulang dari data mentah)
# =====================================================================
def rekonstruksi(folder):
    rows = baca_csv(os.path.join(folder, "raw_signal.csv"))
    t = np.array([fnum(r["t_detik"]) for r in rows])
    rgb = np.array([[fnum(r["R_w"]), fnum(r["G_w"]), fnum(r["B_w"])] for r in rows])
    rois = {
        "dahi": np.array([[fnum(r["R_dahi"]), fnum(r["G_dahi"]), fnum(r["B_dahi"])] for r in rows]),
        "pipi_kiri": np.array([[fnum(r["R_kiri"]), fnum(r["G_kiri"]), fnum(r["B_kiri"])] for r in rows]),
        "pipi_kanan": np.array([[fnum(r["R_kanan"]), fnum(r["G_kanan"]), fnum(r["B_kanan"])] for r in rows]),
    }
    res = rc.analyze_window(t, rgb, rois)
    fs = rc.TARGET_FS

    print("=" * 70)
    print(f"REKONSTRUKSI PERHITUNGAN: {folder}")
    print("=" * 70)
    print(f"[0] Data mentah     : {len(t)} frame selama {t[-1] - t[0]:.2f} s -> FPS efektif = "
          f"({len(t)} - 1) / {t[-1] - t[0]:.2f} = {res['fs_eff']:.2f} fps")
    print(f"    Contoh frame #0 : R={rgb[0, 0]:.3f} G={rgb[0, 1]:.3f} B={rgb[0, 2]:.3f}")
    print(f"[1] Interpolasi     : grid seragam 1/{fs:.0f} s -> {len(res['t_uniform'])} sampel")
    mu = res["rgb_mean"]
    print(f"[2] Rata-rata (DC)  : mean R={mu[0]:.3f} G={mu[1]:.3f} B={mu[2]:.3f}")
    c0 = res["rgb_uniform"][0]
    print(f"    Normalisasi #0  : Rn={c0[0]:.3f}/{mu[0]:.3f}={c0[0] / mu[0]:.5f}  Gn={c0[1] / mu[1]:.5f}  Bn={c0[2] / mu[2]:.5f}")
    print("    (catatan: di POS asli, rata-rata dihitung per jendela 1.6 s, bukan seluruh rekaman)")
    l = int(round(rc.POS_WIN_SEC * fs))
    C = res["rgb_uniform"][:l]
    Cn = C / C.mean(axis=0)
    s1, s2 = Cn[:, 1] - Cn[:, 2], -2 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]
    a = np.std(s1) / np.std(s2)
    print(f"[3] POS jendela #1  : l = 1.6 s x 30 Hz = {l} sampel")
    print(f"    S1[0]=Gn-Bn={s1[0]:.6f}   S2[0]=-2Rn+Gn+Bn={s2[0]:.6f}")
    print(f"    alpha = std(S1)/std(S2) = {np.std(s1):.6f}/{np.std(s2):.6f} = {a:.4f}")
    print(f"    h[0] = S1[0] + alpha*S2[0] = {s1[0] + a * s2[0]:.6f}  (lalu dikurangi mean jendela & di-overlap-add)")
    print("[4] Band-pass       : Butterworth orde 3, 0.75-3.0 Hz (45-180 BPM), filtfilt (tanpa geser fase)")
    print(f"[5] Spektrum Welch  : puncak f = {res['bpm'] / 60:.4f} Hz  ->  BPM = f x 60 = {res['bpm']:.2f}")
    print(f"    SNR (de Haan)   : {res['snr_db']:.2f} dB")
    if res["bpm_ibi"]:
        print(f"[6] IBI             : {len(res['peaks'])} puncak, median IBI = {np.median(res['ibi']):.4f} s "
              f"-> 60/{np.median(res['ibi']):.4f} = {res['bpm_ibi']:.2f} BPM")
    print(f"[7] Pembanding      : CHROM = {res['bpm_chrom']:.2f}  GREEN = {res['bpm_green']:.2f}")
    print(f"[8] Forensik        : korelasi ROI = {res['roi_corr']:.3f}  BPM per ROI = "
          + ", ".join(f"{k} {v:.1f}" for k, v in res["roi_bpm"].items())
          + f"  (selisih {res['roi_bpm_spread']:.1f})")
    live, alasan = rc.classify_liveness(res)
    print(f"    Keputusan       : {'LIVE' if live else 'SPOOF'}  {alasan}")

    ref = None
    js = os.path.join(folder, "ringkasan.json")
    if os.path.exists(js):
        import json
        with open(js, encoding="utf-8") as f:
            d = json.load(f)
        ref = fnum(d.get("BPM_Oksimeter"))
        if ref > 0:
            print(f"[9] Oksimeter       : {ref:.0f} BPM -> error = {res['bpm'] - ref:+.2f} BPM")
            print(f"    BPM tersimpan saat live = {d.get('BPM_rPPG_POS')}  (harus sama dengan hasil rekonstruksi)")
        else:
            ref = None
    out = os.path.join(folder, "rekonstruksi_pipeline.png")
    rc.plot_pipeline(res, out, judul=f"Rekonstruksi offline: {os.path.basename(folder)}", bpm_ref=ref)
    print(f"\nGrafik: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE, "data_uji_rppg_v2.csv"))
    ap.add_argument("--out", default=os.path.join(BASE, "laporan_validasi"))
    ap.add_argument("--rekonstruksi", default="", help="folder sesi di hasil_sesi/")
    a = ap.parse_args()
    if a.rekonstruksi:
        rekonstruksi(a.rekonstruksi)
    else:
        if not os.path.exists(a.csv):
            sys.exit(f"File {a.csv} belum ada. Jalankan rppgforensik_v2.py dulu.")
        rekap(a.csv, a.out)
