import requests
from PIL import Image
import io

img = Image.new('RGB', (100, 100), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

print("Sending request...")
try:
    response = requests.post("http://127.0.0.1:8000/api/analyze", files={"image": ("test.png", img_bytes, "image/png")})
    print("Status Code:", response.status_code)
    print("Response:", response.text[:200])
except Exception as e:
    print("Error:", e)
