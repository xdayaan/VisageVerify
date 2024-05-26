from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS extension
import face_recognition
import requests
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

data = [
    ["Elon Musk", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Elon_Musk_Colorado_2022_%28cropped2%29.jpg/640px-Elon_Musk_Colorado_2022_%28cropped2%29.jpg"],
    ["Barack Obama", "https://cdn.britannica.com/44/127844-050-33AB565F/Barack-Obama-2009.jpg"],
    ["Jeff Bezos", "https://cdn.britannica.com/56/199056-050-CCC44482/Jeff-Bezos-2017.jpg"],
    ["Bill Gates", "https://imageio.forbes.com/specials-images/imageserve/62d599ede3ff49f348f9b9b4/0x0.jpg?format=jpg&crop=821,821,x155,y340,safe&height=416&width=416&fit=bounds"],
    ["Lionel Messi", "https://fcb-abj-pre.s3.amazonaws.com/img/jugadors/MESSI.jpg"],
    ["Mark Zuckerberg", "https://cdn.britannica.com/99/236599-050-1199AD2C/Mark-Zuckerberg-2019.jpg"]
]

known_encodings = []
known_names = []
known_urls = []

# Precompute known encodings and URLs
for name, url in data:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)
    if len(encodings) > 0:
        known_encodings.append(encodings[0])
        known_names.append(name)
        known_urls.append(url)

print("Model trained")

@app.route('/identify', methods=['POST'])
def identify_faces():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is missing'}), 400

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)
    unknown_encodings = face_recognition.face_encodings(img_array)
    matches = []
    for unknown_encoding in unknown_encodings:
        face_info = []
        for i, known_encoding in enumerate(known_encodings):
            match = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if match[0]:
                face_info.append({'name': known_names[i], 'url': known_urls[i]})
        matches.append(face_info)
    return jsonify(matches)

if __name__ == '__main__':
    app.run(debug=True)
