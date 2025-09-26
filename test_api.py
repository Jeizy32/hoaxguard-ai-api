import requests

# URL API lokal (kalau udah deploy ke Railway, ganti dengan URL Railway)
url = "https://hoaxguard-ai-api-production.up.railway.app"

while True:
    teks = input("\nMasukkan teks berita (atau ketik 'exit' untuk keluar): ")
    if teks.lower() == "exit":
        break

    data = {"text": teks}
    try:
        response = requests.post(url, json=data)
        print("Status code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", e)
