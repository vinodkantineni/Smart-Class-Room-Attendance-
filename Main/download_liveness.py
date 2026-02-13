
import os
import requests

# URL = "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
# Use the correct URL that is likely to work
URL = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

SAVE_DIR = "Main/models"
SAVE_PATH = os.path.join(SAVE_DIR, "liveness.pth")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"Downloading model from {URL}...")
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(URL, headers=headers, stream=True)
    if response.status_code == 200:
        with open(SAVE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download complete: {SAVE_PATH}")
    else:
        print(f"❌ Failed to download. Status: {response.status_code}")
except Exception as e:
    print(f"❌ Error downloading: {e}")
