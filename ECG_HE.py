"""
ECG R-Peak-Erkennung und Herzfrequenz-Analyse
================================================

Erkennt R-Zacken in EKG-Signalen (CSV oder MATLAB .mat) und plottet
das gefilterte Signal mit markierten R-Peaks sowie den zeitlichen
Verlauf der Herzfrequenz (BPM).

Abhängigkeiten:
    pip install numpy scipy matplotlib

Beispielaufruf:
    python ecg_heart_rate_analyzer.py --file ekg.csv --fs 250 --column ecg
    python ecg_heart_rate_analyzer.py --file ekg.mat --fs 500 --column signal
    python ecg_heart_rate_analyzer.py --demo   # erzeugt synthetisches EKG
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


@dataclass
class ECGAnalyzer:
    """Kapselt den kompletten Pipeline-Ablauf: Laden -> Filtern -> R-Peaks -> HR."""

    fs: float                      # Abtastrate in Hz
    raw: np.ndarray = None         # Rohsignal
    filtered: np.ndarray = None    # Bandpass-gefiltertes Signal
    r_peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    rr_intervals: np.ndarray = field(default_factory=lambda: np.array([]))
    heart_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    hr_times: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---------- Datenimport ----------

    @classmethod
    def from_csv(cls, path: str | Path, fs: float, column: str | int = 0) -> "ECGAnalyzer":
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.dtype.names and isinstance(column, str) and column in data.dtype.names:
            raw = np.asarray(data[column], dtype=float)
        else:
            raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=int(column) if isinstance(column, int) else 0)
        analyzer = cls(fs=fs)
        analyzer.raw = raw.astype(float)
        return analyzer

    @classmethod
    def from_mat(cls, path: str | Path, fs: float, column: str) -> "ECGAnalyzer":
        from scipy.io import loadmat
        mat = loadmat(path)
        if column not in mat:
            candidates = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"Variable '{column}' nicht in .mat gefunden. Verfügbar: {candidates}")
        raw = np.asarray(mat[column]).squeeze().astype(float)
        analyzer = cls(fs=fs)
        analyzer.raw = raw
        return analyzer

    @classmethod
    def demo_signal(cls, fs: float = 250.0, duration_s: float = 30.0, bpm: float = 72.0,
                     noise_std: float = 0.02, seed: int = 0) -> "ECGAnalyzer":
        """Erzeugt ein synthetisches EKG-artiges Signal zu Testzwecken."""
        rng = np.random.default_rng(seed)
        n = int(fs * duration_s)
        t = np.arange(n) / fs
        rr = 60.0 / bpm
        raw = np.zeros(n)
        beat_time = rr / 2
        while beat_time < duration_s:
            center = beat_time + rng.normal(0, 0.01)  # leichte Variabilität
            idx = int(center * fs)
            width = int(0.02 * fs)
            for k in range(-width, width):
                if 0 <= idx + k < n:
                    raw[idx + k] += np.exp(-0.5 * (k / (0.006 * fs)) ** 2)
            beat_time += rr
        raw += 0.05 * np.sin(2 * np.pi * 0.3 * t)      # Baseline-Wander
        raw += noise_std * rng.standard_normal(n)       # Messrauschen
        analyzer = cls(fs=fs)
        analyzer.raw = raw
        return analyzer

    # ---------- Vorverarbeitung ----------

    def bandpass_filter(self, lowcut: float = 0.5, highcut: float = 40.0, order: int = 3) -> np.ndarray:
        """Butterworth-Bandpass zur Rauschunterdrückung und Baseline-Korrektur."""
        if self.raw is None:
            raise RuntimeError("Kein Rohsignal geladen.")
        nyq = 0.5 * self.fs
        b, a = sig.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        self.filtered = sig.filtfilt(b, a, self.raw)
        return self.filtered

    # ---------- R-Peak-Erkennung ----------

    def detect_r_peaks(self, min_rr_s: float = 0.3) -> np.ndarray:
        """
        Pan-Tompkins-ähnlicher Ansatz: Ableitung -> Quadrieren -> gleitendes
        Fenster-Integral -> Peak-Suche mit adaptiver Höhenschwelle und
        physiologischer Mindestabstand (Refraktärzeit).
        """
        if self.filtered is None:
            self.bandpass_filter()

        diff = np.gradient(self.filtered) * self.fs
        squared = diff ** 2

        window = max(1, int(0.15 * self.fs))
        integrated = np.convolve(squared, np.ones(window) / window, mode="same")

        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        distance = max(1, int(min_rr_s * self.fs))

        peaks, _ = sig.find_peaks(integrated, height=threshold, distance=distance)

        # Peaks vom Integrationssignal auf die tatsächlichen R-Zacken im
        # gefilterten Signal zurück-mappen (lokales Maximum in Nachbarschaft)
        refined = []
        search_radius = max(1, int(0.05 * self.fs))
        for p in peaks:
            lo, hi = max(0, p - search_radius), min(len(self.filtered), p + search_radius)
            local_idx = lo + np.argmax(self.filtered[lo:hi])
            refined.append(local_idx)

        self.r_peaks = np.unique(np.array(refined, dtype=int))
        return self.r_peaks

    # ---------- Herzfrequenz ----------

    def compute_heart_rate(self) -> tuple[np.ndarray, np.ndarray]:
        """Berechnet RR-Intervalle und daraus die momentane Herzfrequenz (BPM)."""
        if len(self.r_peaks) < 2:
            raise RuntimeError("Zu wenige R-Peaks für eine HR-Berechnung erkannt.")

        peak_times = self.r_peaks / self.fs
        self.rr_intervals = np.diff(peak_times)
        self.heart_rate = 60.0 / self.rr_intervals
        self.hr_times = peak_times[1:]  # HR-Wert gilt am zweiten Peak jedes Intervalls
        return self.hr_times, self.heart_rate

    # ---------- Kennzahlen ----------

    def summary(self) -> dict:
        if len(self.heart_rate) == 0:
            self.compute_heart_rate()
        return {
            "n_peaks": len(self.r_peaks),
            "mean_bpm": float(np.mean(self.heart_rate)),
            "min_bpm": float(np.min(self.heart_rate)),
            "max_bpm": float(np.max(self.heart_rate)),
            "sdnn_ms": float(np.std(self.rr_intervals) * 1000),  # HRV-Kennzahl
        }

    # ---------- Visualisierung ----------

    def plot(self, save_path: str | Path | None = None) -> None:
        if self.filtered is None:
            self.bandpass_filter()
        if len(self.r_peaks) == 0:
            self.detect_r_peaks()
        if len(self.heart_rate) == 0:
            self.compute_heart_rate()

        t = np.arange(len(self.filtered)) / self.fs

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

        axes[0].plot(t, self.filtered, linewidth=0.8, label="Gefiltertes EKG")
        axes[0].plot(self.r_peaks / self.fs, self.filtered[self.r_peaks],
                     "ro", markersize=5, label="R-Peaks")
        axes[0].set_title("EKG-Signal mit erkannten R-Peaks")
        axes[0].set_xlabel("Zeit (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.hr_times, self.heart_rate, marker="o", markersize=3,
                     linewidth=1, color="darkred")
        axes[1].set_title("Herzfrequenz über Zeit")
        axes[1].set_xlabel("Zeit (s)")
        axes[1].set_ylabel("Herzfrequenz (BPM)")
        axes[1].grid(alpha=0.3)

        s = self.summary()
        fig.suptitle(
            f"Ø {s['mean_bpm']:.1f} BPM  |  min {s['min_bpm']:.1f}  |  "
            f"max {s['max_bpm']:.1f}  |  SDNN {s['sdnn_ms']:.1f} ms",
            fontsize=10,
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Plot gespeichert unter: {save_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="R-Peak-Erkennung und Herzfrequenz-Analyse für EKG-Daten")
    parser.add_argument("--file", type=str, help="Pfad zur CSV- oder .mat-Datei")
    parser.add_argument("--fs", type=float, default=250.0, help="Abtastrate in Hz (Standard: 250)")
    parser.add_argument("--column", type=str, default="ecg",
                         help="Spaltenname (CSV) bzw. Variablenname (.mat) des Signals")
    parser.add_argument("--demo", action="store_true", help="Synthetisches EKG statt Datei verwenden")
    parser.add_argument("--save", type=str, default=None, help="Optional: Pfad zum Speichern des Plots (PNG)")
    args = parser.parse_args()

    if args.demo or not args.file:
        print("Kein Datenfile angegeben (oder --demo gesetzt) -> verwende synthetisches Testsignal.")
        analyzer = ECGAnalyzer.demo_signal(fs=args.fs)
    else:
        path = Path(args.file)
        if path.suffix.lower() == ".mat":
            analyzer = ECGAnalyzer.from_mat(path, fs=args.fs, column=args.column)
        else:
            analyzer = ECGAnalyzer.from_csv(path, fs=args.fs, column=args.column)

    analyzer.bandpass_filter()
    analyzer.detect_r_peaks()
    analyzer.compute_heart_rate()

    s = analyzer.summary()
    print("\n--- Zusammenfassung ---")
    for k, v in s.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    analyzer.plot(save_path=args.save)


if __name__ == "__main__":
    main()
