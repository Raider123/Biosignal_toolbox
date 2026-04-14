import numpy as np
from pathlib import Path

# Versuche, die Motion-Bibliothek zu laden
try:
    from biosignal_toolbox.motion_lib import MotionData
except ImportError:
    print("Fehler: 'biosignal_toolbox' nicht gefunden. Bitte sicherstellen, dass sie installiert ist.")
    exit()

# --- KONFIGURATION FÜR SUBJECT 1 ---
CONFIG = {
    # Pfade anpassen falls nötig
    "data_path": "data/jte/quali/WW06D/tsv/",
    "filename": "24_07_2025_WW06D_1838g_complex_2.tsv",

    # Subjekt-Parameter aus der Tabelle (Subject 1)
    "body_weight_kg": 93,  # Angepasst an Bild (Subject 1)
    "subject_sex": "male",  # Annahme basierend auf Originalcode

    # Handlänge muss oft manuell gemessen werden.
    # Im Originalcode war sie 113 mm. Falls du einen anderen Wert für Subject 1 hast, hier ändern:
    "hand_length_mm": 115
}


def main():
    print(f"--- Starte Analyse für Datei: {CONFIG['filename']} ---")
    print(f"Gewicht: {CONFIG['body_weight_kg']} kg | Handlänge Input: {CONFIG['hand_length_mm']} mm")
    print("-" * 50)

    # 1. Motion Data laden
    qualisys_data = MotionData(
        data_path=CONFIG["data_path"],
        filename=CONFIG["filename"]
    )

    # WICHTIG: Die Indizes der Marker finden
    # (Namen müssen mit denen in der TSV übereinstimmen, meistens standardisiert)
    try:
        sr_idx = qualisys_data.getChannelIndex(joint_name='shoulder', hand='right')
        er_idx = qualisys_data.getChannelIndex(joint_name='elbow', hand='right')
        wr_idx = qualisys_data.getChannelIndex(joint_name='wrist', hand='right')
    except Exception as e:
        print(f"Fehler beim Finden der Gelenk-Marker: {e}")
        print("Überprüfe, ob die Gelenknamen (shoulder, elbow, wrist) in der Datei stimmen.")
        return

    # ---------------------------------------------------------
    # BERECHNUNG DER LÄNGEN (aus Mocap-Daten)
    # ---------------------------------------------------------

    # 1. Oberarm (Upper Arm) Länge: Schulter -> Ellbogen
    ua_dist_mm = qualisys_data.getEuclideanDistance_mm(joint_idx1=sr_idx, joint_idx2=er_idx)
    ua_length_mm = np.mean(ua_dist_mm)  # Mittelwert über die Zeit
    ua_length_m = ua_length_mm / 1000.0

    # 2. Unterarm (Forearm) Länge: Ellbogen -> Handgelenk
    fa_dist_mm = qualisys_data.getEuclideanDistance_mm(joint_idx1=er_idx, joint_idx2=wr_idx)
    fa_length_mm = np.mean(fa_dist_mm)
    fa_length_m = fa_length_mm / 1000.0

    # 3. Handlänge (Hand Length)
    # Wird meist manuell gemessen und als Parameter übergeben, da Marker an der Fingerspitze oft fehlen.
    # Wir nehmen den konfigurierten Wert.
    hand_length_m = CONFIG["hand_length_mm"] / 1000.0

    # ---------------------------------------------------------
    # BERECHNUNG DER SCHWERPUNKTE (COM)
    # ---------------------------------------------------------
    # Die Methode getCOM_percent gibt zurück, bei wie viel % der Segmentlänge der COM liegt (z.B. nach Winter/Dempster)

    # Upper Arm COM (von der Schulter aus)
    ua_com_perc = qualisys_data.getCOM_percent(segment_name="upperarm", subject_biological_sex=CONFIG["subject_sex"])
    ua_com_len_m = (ua_com_perc / 100.0) * ua_length_m

    # Forearm COM (vom Ellbogen aus)
    fa_com_perc = qualisys_data.getCOM_percent(segment_name="forearm", subject_biological_sex=CONFIG["subject_sex"])
    fa_com_len_m = (fa_com_perc / 100.0) * fa_length_m

    # Hand COM (vom Handgelenk aus)
    hand_com_perc = qualisys_data.getCOM_percent(segment_name="hand", subject_biological_sex=CONFIG["subject_sex"])
    hand_com_len_m = (hand_com_perc / 100.0) * hand_length_m

    # ---------------------------------------------------------
    # AUSGABE FÜR DIE TABELLE
    # ---------------------------------------------------------
    print("\nERGEBNISSE FÜR TABELLE (Subject 1):")
    print(f"{'Characteristic':<40} | {'Wert (m)':<15} | {'Original (mm)'}")
    print("-" * 70)

    # Zeilen passend zu deinem Bild
    print(f"{'Upper arm length':<40} | {ua_length_m:.3f} m        | ({ua_length_mm:.1f} mm)")
    print(f"{'Forearm length':<40} | {fa_length_m:.3f} m        | ({fa_length_mm:.1f} mm)")
    print(f"{'Hand length (Input)':<40} | {hand_length_m:.3f} m        | ({CONFIG['hand_length_mm']} mm)")
    print("-" * 70)
    print(f"{'Upper arm COM length (from shoulder)':<40} | {ua_com_len_m:.3f} m        | ({ua_com_perc:.1f} %)")
    print(f"{'Forearm COM length (from elbow)':<40} | {fa_com_len_m:.3f} m        | ({fa_com_perc:.1f} %)")
    print(f"{'Hand COM length (from wrist)':<40} | {hand_com_len_m:.3f} m        | ({hand_com_perc:.1f} %)")
    print("-" * 70)


if __name__ == "__main__":
    main()