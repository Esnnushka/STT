# STT
Функции
Загрузка аудиофайлов в различных форматах.
Распознавание речи с использованием модели Whisper.
Ограничение длительности аудиофайла перед обработкой (по умолчанию 30 секунд).
Отображение текста и времени обработки на веб-странице.


🚀 Установка и запуск

Клонировать репозиторий:
git clone https://github.com/Esnnushka/STT.git
cd STT

Установить зависимости
pip install -r requirements.txt

Запустить сервер:
python STT.py
Сервер будет доступен по адресу: http://127.0.0.1:5000

📂 Структура проекта
STT/
│
├── STT.py                # Главный файл приложения (Flask)
├── requirements.txt      # Список зависимостей
├── uploads/              # Папка для загруженных файлов
├── processed/            # Папка для временно обработанных файлов
├── templates/            # Шаблоны для рендеринга HTML
│   └── index.html        # Основной HTML файл
│   └── result.html       # Результирующий HTML файл
└── README.md             # Документация проекта

🔧 Используемые технологии
Flask — веб-фреймворк для создания веб-приложений.
Whisper — модель для распознавания речи от OpenAI.
Pydub — библиотека для работы с аудиофайлами.
HTML/CSS — фронтенд для взаимодействия с пользователем.

📝 Как работает приложение?
Пользователь загружает аудиофайл в формате, поддерживаемом браузером (например, mp3, wav).
На сервере файл обрабатывается с помощью библиотеки Pydub, чтобы ограничить длительность аудио (по умолчанию 30 секунд).
Затем аудиофайл передается в модель Whisper, которая распознает речь и конвертирует ее в текст.
После обработки результат (текст и время обработки) отправляется обратно на веб-страницу, где отображается пользователю.

⚙️ Зависимости
Для работы проекта необходимы следующие библиотеки:

Flask — веб-фреймворк для Python
Whisper — модель для распознавания речи
Pydub — инструмент для обработки аудиофайлов
FFmpeg — для обработки аудио в различных форматах (необходим для работы с Pydub)

FFmpeg необходимо установить отдельно:
Для Windows скачайте и установите FFmpeg с официального сайта: FFmpeg.

Скачать FFmpeg https://ffmpeg.org/download.html (выбери версию для Windows).
Распаковать архив и добавь путь к ffmpeg.exe в переменную окружения PATH.
Проверь установку командой ffmpeg -version

При первом запуске программы будет произведена установка модели для распознания речи. Вес модели ~1.9 Gb

Тесты проводились на видеокарте 1660 Super + I5 9600K + 16 Gb ОЗУ.
Время обработки видео/аудио дорожек предоставленных в кейсах:
Задание 1.mp4: 
    30 сек ~ 50 сек.
    60 сек ~ 2 мин. 
    600 сек ~ 5 мин.  

Задание 2.m4a: 
    30 сек ~ 52 сек.
    60 сек ~ 90 секунд. 
    120 сек ~ 150 сек.

Загружаемые файлы не должны содержать латиницу и пробелы.