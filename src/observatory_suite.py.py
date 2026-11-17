#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LANDSTUHL RADIO OBSERVATORY - ASTROPHYSICAL SUITE v14.0
========================================================
Das "Wohnzimmer-Observatorium" für echte 21-cm-Radioastronomie.

Highlights v14.0:
  * NEU: Absturz-sicherer WAV-Parser. Rettet auch abgebrochene 
         GQRX-Aufnahmen, bei denen der Header fehlt/zerschossen ist.
  * Modus 4 sucht gezielt nach Wasserstoffwolken (21-cm-Fenster).
  * Modus 5 zoomt automatisch auf den gemessenen Sternzeit-Bereich (LST).
  * OOM-sicher: IQSource (memmap) + blockweise FFT-Verarbeitung.
"""

import os
import re
import glob
import wave
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless; für interaktiv: "TkAgg"
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =====================================================================
# KONSTANTEN
# =====================================================================
C_KMS       = 299792.458            # Lichtgeschwindigkeit [km/s]
F_NULL_HZ   = 1420405751.768        # 21cm-Ruhefrequenz [Hz]
DEFAULT_F_C = 1420175100.0          # Fallback-Mittenfrequenz [Hz]
HOME_DIR    = os.path.expanduser("~/Weltraumdaten")
OUTPUT_DIR  = os.path.expanduser("~/Schreibtisch")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# WAV-HEADER-PARSER (Crash-sicher & RIFF-Padding-sicher)
# =====================================================================
def wav_data_offset(pfad):
    """
    Sucht den 'data'-Chunk. Wenn die Datei zerschossen ist (z.B. weil
    GQRX abgestürzt ist), fällt die Funktion auf Standard-Werte zurück.
    """
    dateigroesse = os.path.getsize(pfad)
    try:
        with open(pfad, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                # Kein RIFF-WAVE Header -> Blind ab Byte 44 lesen
                return 44, dateigroesse - 44
                
            pos = 12
            while True:
                f.seek(pos)
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = int.from_bytes(f.read(4), "little")
                if chunk_id == b"data":
                    # Wenn der Header eine 0 oder Müll enthält, nehmen wir den Rest der Datei
                    if chunk_size == 0 or (pos + 8 + chunk_size) > dateigroesse:
                        chunk_size = dateigroesse - (pos + 8)
                    return pos + 8, chunk_size
                pos += 8 + chunk_size + (chunk_size & 1)
    except Exception:
        pass
        
    # Fallback wenn absolut nichts geht
    return 44, dateigroesse - 44


def wav_header_info(pfad):
    try:
        with wave.open(pfad, "rb") as wf:
            return (wf.getframerate(), wf.getnchannels(),
                    wf.getsampwidth(), wf.getnframes())
    except (wave.Error, EOFError):
        # Fallback für zerschossene Dateien
        return 48000, 2, 2, None


def extrahiere_zentralfrequenz(dateiname):
    treffer = re.findall(r"_(\d+(?:\.\d+)?)(?:\.wav)?$",
                         os.path.basename(dateiname))
    if treffer:
        try:
            f = float(treffer[-1])
            if f > 1e6:
                return f
        except ValueError:
            pass
    return DEFAULT_F_C


# =====================================================================
# IQSource: memmap-basiert, lazy komplexe Konvertierung (OOM-sicher)
# =====================================================================
class IQSource:
    def __init__(self, pfad):
        # Versuche, den Header normal zu lesen
        try:
            with wave.open(pfad, "rb") as wf:
                self.n_channels = wf.getnchannels()
                sampwidth       = wf.getsampwidth()
        except wave.Error:
            # NOTFALL: Datei wurde nicht sauber beendet.
            print("  ⚠️ WARNUNG: WAV-Header defekt! Versuche Rohdaten zu retten...")
            self.n_channels = 2
            sampwidth       = 2

        if sampwidth != 2:
            raise ValueError(f"Erwartet 16-bit, gefunden {sampwidth*8}-bit")

        offset, data_bytes = wav_data_offset(pfad)
        
        frame_bytes    = self.n_channels * 2
        n_ganze_frames = data_bytes // frame_bytes
        n_werte        = n_ganze_frames * self.n_channels

        if n_werte == 0:
            raise ValueError("Datei enthält keine lesbaren IQ-Daten.")

        self._mm = np.memmap(pfad, dtype=np.int16, mode="r",
                             offset=offset, shape=(n_werte,))
        self.n_complex = n_ganze_frames
        self.pfad      = pfad

    def __len__(self):
        return self.n_complex

    def __getitem__(self, sl):
        if isinstance(sl, slice):
            start, stop, step = sl.indices(self.n_complex)
            if step != 1:
                raise NotImplementedError("Schrittweite != 1 nicht unterstützt")
            n = max(0, stop - start)
            if n == 0:
                return np.zeros(0, dtype=np.complex64)
            roh = self._mm[start * self.n_channels : stop * self.n_channels]
            if self.n_channels >= 2:
                i = roh[0::2].astype(np.float32)
                q = roh[1::2].astype(np.float32)
                return i + 1j * q
            else:
                return roh.astype(np.float32).astype(np.complex64)
        idx = sl if sl >= 0 else self.n_complex + sl
        roh = self._mm[idx * self.n_channels : (idx + 1) * self.n_channels]
        if self.n_channels >= 2:
            return complex(float(roh[0]), float(roh[1]))
        return complex(float(roh[0]), 0.0)


# =====================================================================
# SIGNAL-HILFSFUNKTIONEN
# =====================================================================
def fft_magnitude_db(iq_block):
    if len(iq_block) == 0:
        return np.zeros(0)
    fenster  = np.hanning(len(iq_block)).astype(np.float32)
    spektrum = np.fft.fftshift(np.fft.fft(iq_block * fenster))
    return 20.0 * np.log10(np.abs(spektrum) + 1e-12)


def geschwindigkeitsachse(n_fft, samplerate, f_zentrum, f_null=F_NULL_HZ):
    freq_offsets = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / samplerate))
    echte_freq   = f_zentrum + freq_offsets
    return C_KMS * ((f_null - echte_freq) / f_null)


def berechne_lst(utc_dt, laengengrad_ost=7.57):
    """Berechnet die Local Sidereal Time (Sternzeit) für Landstuhl."""
    y, m = utc_dt.year, utc_dt.month
    d = utc_dt.day + (utc_dt.hour + utc_dt.minute/60 + utc_dt.second/3600) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5

    T = (JD - 2451545.0) / 36525.0
    GST = 6.697374558 + 0.06570982441908 * (JD - 2451545.0) + \
          1.00273790935 * (utc_dt.hour + utc_dt.minute/60 + utc_dt.second/3600)
    GST = GST % 24.0
    LST = (GST + laengengrad_ost / 15.0) % 24.0
    return LST


# =====================================================================
# INTERAKTIVE AUSWAHL
# =====================================================================
def datei_auswahl():
    such_pfad = os.path.join(HOME_DIR, "gqrx_*.wav")
    dateien = sorted(glob.glob(such_pfad), key=os.path.getctime)
    if not dateien:
        print(f"Fehler: Keine GQRX-WAV-Datei in {HOME_DIR} gefunden!")
        return None

    print("\n" + "=" * 70)
    print(" 📡  LANDSTUHL RADIO OBSERVATORY - ASTROPHYSICAL SUITE v14.0")
    print("=" * 70)
    for idx, pfad in enumerate(dateien):
        mb = os.path.getsize(pfad) / (1024 * 1024)
        print(f"[{idx}] {os.path.basename(pfad)} ({mb:.1f} MB)")

    print("\nWählen Sie das Datenpaket für die Analyse (Index):")
    try:
        auswahl = int(input("> "))
        return dateien[auswahl]
    except (ValueError, IndexError):
        print("Ungültige Auswahl!")
        return None


def modus_auswahl():
    print("\n" + "-" * 70)
    print(" 🛰️  CHOOSE ANALYTICAL MODE:")
    print(" [1] 2D-Radio-Himmelskartierung (Plasma-Heatmap)")
    print(" [2] 21cm Intensitätsprofil (Baseline-bereinigt)")
    print(" [3] Kosmologische Massen-Analyse (Simulation)")
    print(" [4] ECHTE Wasserstoff-Wolken-Karte (Peak-Match)")
    print(" [5] ECHTE differentielle Rotation (Doppler-Analyse & Sternzeit)")
    print("-" * 70)
    try:
        return int(input("Modus wählen > "))
    except ValueError:
        print("Ungültiger Modus!")
        return None


# =====================================================================
# MODUS 1: 2D-HIMMELSKARTIERUNG
# =====================================================================
def modus_1(iq, samplerate, f_zentrum):
    print("[MODUS 1] Starte zeitliche 21cm-Intensitätskartierung...")
    block_sekunden  = 10
    samples_pro_blk = int(block_sekunden * samplerate)
    total_blocks    = len(iq) // samples_pro_blk
    if total_blocks == 0: return

    linien_offsets_hz = (F_NULL_HZ * (1 - np.array([-200.0, 200.0]) / C_KMS) - f_zentrum)
    intensitaeten, zeit_min = [], []
    
    for b in range(total_blocks):
        if b % 50 == 0: print(f"  ... Block {b}/{total_blocks}", flush=True)
        block  = iq[b * samples_pro_blk : (b + 1) * samples_pro_blk]
        mag_db = fft_magnitude_db(block)
        freqs  = np.fft.fftshift(np.fft.fftfreq(len(block), d=1.0 / samplerate))
        maske  = (freqs >= linien_offsets_hz.min()) & (freqs <= linien_offsets_hz.max())
        if np.any(maske):
            intensitaeten.append(float(np.max(mag_db[maske])))
            zeit_min.append(b * block_sekunden / 60.0)

    if not intensitaeten: return
    karte = np.array([intensitaeten])
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(karte, aspect="auto", cmap="plasma", extent=[0, zeit_min[-1], 0, 1])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2).set_label("Intensität (dB)")
    ax.set_title("21cm Radio-Himmelskartierung – Zeitreihe")
    ziel = os.path.join(OUTPUT_DIR, "AllInOne_2D_Himmelskarte.png")
    plt.savefig(ziel, bbox_inches="tight"); plt.close()
    print(f"  -> gespeichert: {ziel}")


# =====================================================================
# MODUS 2: INTENSITÄTSPROFIL
# =====================================================================
def modus_2(iq, samplerate, f_zentrum):
    print("[MODUS 2] Akkumuliere Spektrum...")
    block_groesse = 10 * samplerate
    n_bloecke     = len(iq) // block_groesse
    if n_bloecke == 0: return

    akkum = np.zeros(block_groesse, dtype=np.complex128)
    for i in range(n_bloecke):
        if i % 50 == 0: print(f"  ... Block {i}/{n_bloecke}", flush=True)
        block   = iq[i * block_groesse : (i + 1) * block_groesse]
        fenster = np.hanning(len(block)).astype(np.float32)
        akkum  += np.fft.fftshift(np.fft.fft(block * fenster))
        
    mag_db = 20.0 * np.log10(np.abs(akkum / n_bloecke) + 1e-12)
    v_achse = geschwindigkeitsachse(block_groesse, samplerate, f_zentrum)
    signal_db = mag_db - np.median(mag_db)
    idx_peak  = int(np.argmax(signal_db))
    v_peak    = v_achse[idx_peak]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(v_achse, signal_db, color="#2ca02c", lw=1.2)
    ax.fill_between(v_achse, signal_db, color="#2ca02c", alpha=0.3)
    ax.axvline(0, color="gray", ls=":", label="Ruhe (v=0)")
    ax.set_xlim(v_peak - 100, v_peak + 100)
    ax.set_title("21cm Intensitätsprofil (Baseline-bereinigt)")
    ax.set_xlabel("Radialgeschwindigkeit (km/s)")
    ziel = os.path.join(OUTPUT_DIR, "AllInOne_Intensitaetsprofil.png")
    plt.savefig(ziel, bbox_inches="tight"); plt.close()
    print(f"  Peak bei v = {v_peak:+.1f} km/s  -> gespeichert: {ziel}")


# =====================================================================
# MODUS 3: KOSMOLOGISCHE MASSEN-ANALYSE (SIMULATION)
# =====================================================================
def modus_3(iq, samplerate, f_zentrum):
    print("[MODUS 3] Starte kosmologisches Massenmodul (SIMULATION)...")
    unterblock = 1 << 18
    bloecke_pro_stunde = max(1, int(samplerate * 3600 // unterblock))
    samples_pro_stunde = bloecke_pro_stunde * unterblock
    gesamt_stunden = max(1, len(iq) // samples_pro_stunde)
    faktor_abfall  = 0.12 if gesamt_stunden > 1 else 0.3

    v_achse = geschwindigkeitsachse(unterblock, samplerate, f_zentrum)
    fenster = np.hanning(unterblock).astype(np.float32)
    stunden, v_real_l, v_newton_l, prozent_dm, fehler_dm = [], [], [], [], []

    for stunde in range(gesamt_stunden):
        print(f"  ... Sektor {stunde + 1}/{gesamt_stunden}", flush=True)
        start = stunde * samples_pro_stunde
        akkum = np.zeros(unterblock, dtype=np.complex128)
        n_ok  = 0
        for k in range(bloecke_pro_stunde):
            offset = start + k * unterblock
            block  = iq[offset : offset + unterblock]
            if len(block) < unterblock: break
            akkum += np.fft.fftshift(np.fft.fft(block * fenster))
            n_ok  += 1
        if n_ok == 0: continue

        mag_db   = 20.0 * np.log10(np.abs(akkum / n_ok) + 1e-12)
        idx_peak = int(np.argmax(mag_db))
        v_real   = abs(float(v_achse[idx_peak]))
        v_newton = v_real / np.sqrt(1 + faktor_abfall * stunde)
        m_sicht, m_total = v_newton**2, v_real**2
        dm_anteil = max(0.0, ((m_total - m_sicht) / m_total) * 100.0) if m_total > 0 else 0.0

        stunden.append(stunde)
        v_real_l.append(v_real)
        v_newton_l.append(v_newton)
        prozent_dm.append(dm_anteil)
        fehler_dm.append(min(4.0, (np.std(mag_db) / (np.max(mag_db) + 1e-9)) * 100 * (1.5 / (stunde + 1))))

    if not stunden: return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(stunden, v_real_l, "o-", color="#d62728", lw=2.5, label="Messung")
    ax1.plot(stunden, v_newton_l, "--", color="#7f7f7f", lw=2, label="Kepler-Erwartung")
    ax1.set_title("1. Radiale Flussdichte der Wasserstofflinie")
    ax1.set_ylabel("Geschwindigkeit (km/s)")
    ax1.legend(); ax1.grid(ls="--", alpha=0.4)

    ax2.bar(stunden, prozent_dm, color="#2ca02c", alpha=0.7, width=0.4, label="DM (Sim.)")
    ax2.errorbar(stunden, prozent_dm, yerr=fehler_dm, fmt="none", ecolor="#d62728", capsize=4)
    ax2.axhline(50, color="#7f7f7f", ls=":")
    ax2.set_title("2. Massenanteil (SIMULATION)")
    ax2.set_ylabel("Massenanteil (%)"); ax2.set_ylim(0, 110)
    ax2.legend(); ax2.grid(ls="--", alpha=0.4)

    plt.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "AllInOne_Kosmologie_Maximum.png")
    plt.savefig(ziel, bbox_inches="tight"); plt.close()
    print(f"  -> gespeichert: {ziel}")


# =====================================================================
# MODUS 4: ECHTE WASSERSTOFF-WOLKEN-KARTE
# =====================================================================
def modus_4(iq, samplerate, f_zentrum):
    print("[MODUS 4] Starte Wasserstoff-Wolken-Kartierung...")
    block_sekunden  = 60
    samples_pro_blk = int(block_sekunden * samplerate)
    total_blocks    = len(iq) // samples_pro_blk
    if total_blocks == 0: return

    start_ra_h = 17.32
    ra_achse, leistung = [], []

    for b in range(total_blocks):
        if b % 50 == 0: print(f"  ... Block {b}/{total_blocks}", flush=True)
        block = iq[b * samples_pro_blk : (b + 1) * samples_pro_blk]
        if len(block) < samples_pro_blk: continue
        
        mag_db = fft_magnitude_db(block)
        v_achse_blk = geschwindigkeitsachse(len(block), samplerate, f_zentrum)
        
        linien_fenster = (v_achse_blk > -200) & (v_achse_blk < 200)
        baseline = np.median(mag_db)
        signal_db = mag_db - baseline
        
        if np.any(linien_fenster):
            leistung.append(float(np.max(signal_db[linien_fenster])))
        else:
            leistung.append(0.0)
            
        ra_achse.append((start_ra_h + (b * block_sekunden / 3600.0)) % 24.0)

    if not leistung: return
    leistung = np.array(leistung)
    
    schwelle = np.median(leistung) + 1.5 * np.std(leistung)
    peak_idx, _ = find_peaks(leistung, height=schwelle)
    peak_ras = [ra_achse[i] for i in peak_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ra_achse, leistung, color="#9467bd", lw=2, label="21-cm Wasserstoff-Scan")
    ax.fill_between(ra_achse, leistung, color="#9467bd", alpha=0.15)
    
    for p_ra, p_idx in zip(peak_ras, peak_idx):
        p_hoehe = leistung[p_idx]
        ax.scatter([p_ra], [p_hoehe], color="#d62728", s=80, edgecolors="black", zorder=5)
        ax.text(p_ra + 0.05, p_hoehe + 0.5, "Wolke", fontsize=8, color="#d62728")

    ax.set_title("Echte Wasserstoff-Wolken-Karte (Landstuhl Sky Survey)")
    ax.set_xlabel("Rektaszension (h)")
    ax.set_ylabel("Signalstärke über Rauschen (dB)")
    ax.grid(ls="--", alpha=0.4); ax.legend()
    ziel = os.path.join(OUTPUT_DIR, "Echte_Wasserstoff_Karte.png")
    plt.savefig(ziel, bbox_inches="tight"); plt.close()
    print(f"  -> gespeichert: {ziel}")


# =====================================================================
# MODUS 5: ECHTE DIFFERENTIELLE ROTATION (Doppler & Sternzeit)
# =====================================================================
def modus_5(iq, samplerate, f_zentrum, dateiname):
    print("[MODUS 5] Starte ECHTE differentielle Rotationsmessung...")
    
    try:
        match = re.search(r"gqrx_(\d{8})_(\d{6})", os.path.basename(dateiname))
        if match:
            datum_str, zeit_str = match.groups()
            start_utc = datetime.datetime.strptime(datum_str + zeit_str, "%Y%m%d%H%M%S")
        else:
            start_utc = datetime.datetime.utcnow()
    except Exception:
        start_utc = datetime.datetime.utcnow()

    print(f"  Startzeit (UTC): {start_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    block_sekunden = 300  # 5-Minuten-Blöcke
    samples_pro_blk = int(block_sekunden * samplerate)
    total_blocks = len(iq) // samples_pro_blk
    if total_blocks == 0: return

    print(f"  Datei deckt {total_blocks * block_sekunden / 3600:.2f} h ab ({total_blocks} Blöcke à 5 min).")

    fenster = np.hanning(samples_pro_blk).astype(np.float32)
    v_achse = geschwindigkeitsachse(samples_pro_blk, samplerate, f_zentrum)
    linien_fenster = (v_achse > -150) & (v_achse < 150)

    zeiten_h, lst_h, v_peaks = [], [], []

    for b in range(total_blocks):
        if b % 10 == 0: print(f"  ... Block {b}/{total_blocks}", flush=True)
        block = iq[b * samples_pro_blk : (b + 1) * samples_pro_blk]
        if len(block) < samples_pro_blk: continue

        spektrum = np.fft.fftshift(np.fft.fft(block * fenster))
        mag_db = 20.0 * np.log10(np.abs(spektrum) + 1e-12)
        signal_db = mag_db - np.median(mag_db)

        linien_signal = signal_db[linien_fenster]
        linien_v = v_achse[linien_fenster]
        
        if len(linien_signal) > 0:
            idx_peak = np.argmax(linien_signal)
            v_peak = linien_v[idx_peak]
            
            block_utc = start_utc + datetime.timedelta(seconds=b * block_sekunden + block_sekunden/2)
            lst = berechne_lst(block_utc)
            
            zeiten_h.append(b * block_sekunden / 3600.0)
            lst_h.append(lst)
            v_peaks.append(v_peak)

    if not v_peaks: return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    ax1.plot(zeiten_h, v_peaks, 'o-', color='#1f77b4', lw=2, markersize=4)
    ax1.set_title("Echte 21-cm-Radialgeschwindigkeit über die Beobachtungszeit", fontweight="bold")
    ax1.set_xlabel("Zeit seit Start (Stunden)")
    ax1.set_ylabel("Radialgeschwindigkeit (km/s)")
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.axhline(0, color='gray', ls=':', label='Ruhe (v=0)')
    ax1.legend()

    ax2.plot(lst_h, v_peaks, 's-', color='#d62728', lw=1.5, markersize=4, alpha=0.8)
    ax2.set_title("Differentielle Rotation: Geschwindigkeit vs. Sternzeit (LST)", fontweight="bold")
    ax2.set_xlabel("Local Sidereal Time (Stunden) – Himmelsrichtung")
    ax2.set_ylabel("Radialgeschwindigkeit (km/s)")
    
    # Automatischer Zoom auf den gemessenen Bereich
    ax2.set_xlim(min(lst_h) - 0.5, max(lst_h) + 0.5)
    
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.axhline(0, color='gray', ls=':', label='Ruhe (v=0)')
    ax2.legend()

    plt.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "Echte_Rotationskurve.png")
    plt.savefig(ziel, bbox_inches="tight"); plt.close()
    print(f"  -> gespeichert: {ziel}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    dateiname = datei_auswahl()
    if dateiname is None: return

    modus = modus_auswahl()
    if modus not in (1, 2, 3, 4, 5): return

    print(f"\n[🚀 SYSTEM-CORE] Öffne IQ-Daten aus {os.path.basename(dateiname)} ...")
    try:
        samplerate, n_channels, sampwidth, _ = wav_header_info(dateiname)
        f_zentrum = extrahiere_zentralfrequenz(dateiname)
        print(f"  Samplerate    : {samplerate} Hz")
        print(f"  Kanäle        : {n_channels}")
        print(f"  Mittenfreq.   : {f_zentrum/1e6:.6f} MHz")

        iq = IQSource(dateiname)
        print(f"  Komplexe Samples: {len(iq):,} (~{len(iq)/samplerate/3600:.2f} h)")
    except Exception as e:
        print(f"\n❌ FEHLER beim Laden der Datei: {e}")
        return

    if len(iq) == 0:
        print("Leere Datei – Abbruch."); return

    if   modus == 1: modus_1(iq, samplerate, f_zentrum)
    elif modus == 2: modus_2(iq, samplerate, f_zentrum)
    elif modus == 3: modus_3(iq, samplerate, f_zentrum)
    elif modus == 4: modus_4(iq, samplerate, f_zentrum)
    elif modus == 5: modus_5(iq, samplerate, f_zentrum, dateiname)


if __name__ == "__main__":
    main()