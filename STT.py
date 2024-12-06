import time
import whisper
from pydub import AudioSegment

# Конвертация и нарезка аудиофайла
def preprocess_audio(audio_file, duration_limit=None):
    """
    Подготавливает аудио к обработке:
    - конвертирует аудио в подходящий формат (WAV)
    - нарезает аудио по указанному лимиту времени (в секундах)
    """
    print("Подготовка аудиофайла...")
    audio = AudioSegment.from_file(audio_file)

    if duration_limit:
        print(f"Ограничение времени: {duration_limit} секунд.")
        audio = audio[:duration_limit * 1000]  # Урезаем до лимита (в миллисекундах)

    processed_file = "processed_audio.wav"
    audio.export(processed_file, format="wav")
    print(f"Файл сохранён как {processed_file}.")
    return processed_file

# Функция для вычисления затраченного времени
def get_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    return time.strftime("%Hч %Mм %Sс", time.gmtime(elapsed_time))  # Используем gmtime для правильного расчета

# Распознавание речи с Whisper
def transcribe_audio_whisper(audio_file, model, language="ru"):
    """
    Распознает речь из аудиофайла с использованием модели Whisper.
    """
    start_time = time.time()
    print("Начало распознавания...")
    print(f"Время начала преобразования: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    result = model.transcribe(audio_file, language=language)
    
    end_time = time.time()
    print("Распознавание завершено.")
    print(f"Время окончания преобразования: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    elapsed_time = get_elapsed_time(start_time, end_time)
    print(f"Затраченное время на преобразование: {elapsed_time}")

    return result["text"]

# Пример использования
if __name__ == "__main__":
    input_file = "zadanie2.m4a"  # Укажи путь к твоему аудиофайлу
    duration_limit = 180  # Ограничение времени распознавания (в секундах)

    # Загружаем модель Whisper
    print("Загрузка модели Whisper...")
    model = whisper.load_model("large")  # Можно использовать "tiny", "small", "medium", "large"
    print("Модель загружена.")

    # Предобрабатываем аудио
    processed_audio = preprocess_audio(input_file, duration_limit)

    # Распознаем текст
    recognized_text = transcribe_audio_whisper(processed_audio, model)
    print(f"Распознанный текст:\n{recognized_text}")

    # Сохраняем результат
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(recognized_text)