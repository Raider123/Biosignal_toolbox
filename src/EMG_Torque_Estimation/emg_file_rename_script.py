import os
import re

# Ordner, in dem die TXT-Dateien liegen (anpassen falls nötig)
folder ='F:/SMT_MASTERPROJEKT/biosignal_toolbox/biosignal_toolbox-dev/data/jte/emg/WW06D/'

# Regex: 8 Ziffern (Datum), dann beliebige Ziffern, dann "_BU"
pattern = re.compile(r"^(\d{8})_\d+(_WW.*\.txt)$")

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        match = pattern.match(filename)
        if match:
            new_name = match.group(1) + match.group(2)
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            # Umbenennen
            os.rename(old_path, new_path)
            print(f"✅ {filename} -> {new_name}")
