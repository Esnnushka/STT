import os
import time
import whisper
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment

# Инициализация приложения Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MODEL = whisper.load_model("medium")  # Загружаем модель заранее

# Конвертация и нарезка аудиофайла
def preprocess_audio(audio_file, duration_limit=None):
    audio = AudioSegment.from_file(audio_file)
    if duration_limit:
        audio = audio[:duration_limit * 1000]  # Урезаем до лимита (в миллисекундах)
    processed_file = os.path.join(PROCESSED_FOLDER, "processed_audio.wav")
    audio.export(processed_file, format="wav")
    return processed_file

# Функция для вычисления затраченного времени
def get_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    return time.strftime("%Hч %Mм %Sс", time.gmtime(elapsed_time))

# Распознавание речи с Whisper
def transcribe_audio_whisper(audio_file, model, language="ru"):
    start_time = time.time()
    result = model.transcribe(audio_file, language=language)
    end_time = time.time()
    elapsed_time = get_elapsed_time(start_time, end_time)
    return result["text"], elapsed_time

# Главная страница
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Обработка загрузки и распознавания
@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Файл не загружен."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Файл не выбран."}), 400

        # Сохранение файла
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Ограничение времени (в секундах)
        duration_limit = 120  # Например, 30 секунд

        # Предобработка и распознавание
        processed_audio = preprocess_audio(file_path, duration_limit)
        recognized_text, elapsed_time = transcribe_audio_whisper(processed_audio, MODEL)

        # Удаляем временные файлы
        os.remove(file_path)
        os.remove(processed_audio)

        # Возвращаем результат
        return jsonify({"text": recognized_text, "elapsed_time": elapsed_time})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
