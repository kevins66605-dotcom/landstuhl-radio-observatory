# Landstuhl Radio Observatory – 21-cm Hydrogen Line DSP Suite

Ein vollständig selbstgebautes Amateur-Radioastronomie-Observatorium am Standort **Landstuhl, Deutschland (49,41° N, 7,57° O)** zum Empfang und zur spektralen Auswertung der **21-cm-Wasserstofflinie (HI-Linie bei 1420,40575 MHz)** unserer Milchstraße.

## System-Architektur & Empfangskette

Das Projekt bildet die gesamte Signalkette vom kosmischen Radiosignal bis zur astrophysikalischen Datenauswertung in Python ab:

### 1. Parabolspiegel & Feedhorn (100 % Low-Cost-Eigenbau aus Pappe & Alufolie)
- **Hauptreflektor:** Handelsübliche **Offset-Satellitenschüssel**, bei der der ursprüngliche LNB im Brennpunkt durch das selbstgebaute 1,42-GHz-Feedhorn ersetzt wurde, um maximalen Antennengewinn und hohe Richtwirkung im L-Band zu erzielen.
- **Rechteck-Hohlleiter (Wellenleiter-Speisekasten, Innenmaß 165 mm × 82,5 mm × 160 mm für TE10-Grundmode):**
  - `2× Kurze Seitenwand`: `82,5 mm × 160 mm`
  - `2× Lange Seitenwand`: `(165 mm + 2 × Pappdicke) × 160 mm` (bei 4 mm Pappe: `173 mm × 160 mm`)
  - `1× Rückwand (Kurzschlussdeckel)`: `(165 mm + 2 × Pappdicke) × (82,5 mm + 2 × Pappdicke)` (bei 4 mm Pappe: `173 mm × 90,5 mm`)
- **Pyramiden-Trichter (Horn-Apertur vorne: 400 mm × 300 mm):**
  - `2× Breite Trichterwand (oben & unten)`: Untere Kante `165 mm` (am Kasten), obere Kante `400 mm` (vorne offen), Trapez-Höhe (senkrecht) `414,5 mm`, schräge Seitenkante `431 mm`
  - `2× Schmale Trichterwand (links & rechts)`: Untere Kante `82,5 mm` (am Kasten), passend auf die schräge Seitenkante (`431 mm`) abgestimmt
- **Antennensonde (Monopol):** `λ/4`-Strahler (`ca. 5,2 cm`) im Abstand von `λ_g/4` von der Rückwand, realisiert über einen handelsüblichen **TV-Koaxialstecker** als HF-Durchführung.

### 2. HF-Vorverstärkung & SDR-Empfänger
- **Low-Noise Amplifier (LNA) & Bandpass-Filter:** Rauscharmer Vorverstärker mit integriertem 1420-MHz-Filter direkt an der Antennenspeisung, um das schwache Wasserstoffsignal anzuheben und Außerband-Störungen (Mobilfunk, WLAN, Radar) zu unterdrücken.
- **Empfänger:** **RTL-SDR USB-Dongle** zur Digitalisierung des Basisband-IQ-Signals und Aufzeichnung als Rohdaten-Stream via **GQRX**.

### 3. Signalverarbeitung & Software (`src/observatory_suite.py`)
- **OOM-sicheres Streaming:** Verarbeitung mehrerer Gigabyte großer GQRX-IQ-Aufnahmen (`WAV`) mittels `numpy.memmap` und blockweiser FFT ohne Überlastung des Arbeitsspeichers.
- **Crash-sicherer RIFF/WAV-Parser:** Rekonstruiert auch unvollständige oder abgebrochene SDR-Aufnahmen ohne gültigen Header.
- **Astrophysikalische Auswertung:** Umrechnung der Frequenzverschiebung in radiale Doppler-Geschwindigkeiten (`km/s`) sowie Berechnung der lokalen Sternzeit (**Local Sidereal Time / LST**) für den Standort Landstuhl (`7,57° O`).

## Projektstruktur

- `src/observatory_suite.py` – Python-Analyse-Suite (FFT-Spektralanalyse, Hanning-Fensterung, Doppler-Geschwindigkeitsachse, LST-Berechnung & Matplotlib-Plots)
