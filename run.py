import json
import logging
import uuid
import os
import re
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import aiofiles
import edge_tts
import g4f
import httpx
import mistune
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from langdetect import detect
import scipy as sp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

from bot import send_file, send_files, server_sturtup, server_shutdown

# Настройка логгера
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация переменных и настроек
MAX_QUESTION_HISTORY_SIZE = 1
ADMIN_PASSWORD = 'kita best waifu'

question_history = []

# Путь к папке для синтеза речи
synthesis_path="voice"

# Проверяем, существует ли папка
if not os.path.exists(synthesis_path):
    # Создаем новую папку
    os.makedirs(synthesis_path)
    print(f"Папка {synthesis_path} успешно создана")
else:
    print(f"Папка {synthesis_path} уже существует")


# Функция планировщика
async def scheduled_task():
    print("Задача выполняется каждые 6 часов.")
    
    file_paths = ['data.json', 'logs.json', 'contexts.json', 'admin_questions.json', 'users_questions.json']
    await send_files(file_paths)

# Lifespan (жизненный цикл) приложения FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Приложение стартует...")
    await server_sturtup()

    # Инициализация и настройка планировщика
    scheduler = AsyncIOScheduler()

    # Добавляем задачу в планировщик
    scheduler.add_job(scheduled_task, IntervalTrigger(hours=6)) 
    scheduler.start()

    yield

    print("Приложение завершает работу...")
    scheduler.shutdown()

    file_paths = ['data.json', 'logs.json', 'contexts.json', 'admin_questions.json', 'users_questions.json']
    await send_files(file_paths)
    
    await server_shutdown()


# Инициализация приложения FastAPI с lifespan
app = FastAPI(
    title="Виртуальный помощник",  # Заголовок API
    description="**API для взаимодействия с виртуальным помощником, который отвечает на вопросы и помогает с решением задач.**",
    version="2.0.4",  # Версия API
    contact={
        "name": "Developer ikymuco",
        "email": "xzzlinkzzx@gmail.com",
        "url": "https://t.me/ikymuco"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)


# Настройка CORS
origins = [
    "*", # Разрешить доступ с любого домена
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Разрешённые источники
    allow_credentials=True,  # Разрешить ли отправлять cookies и авторизационные заголовки
    allow_methods=["*"],     # Разрешённые HTTP методы
    allow_headers=["*"],     # Разрешённые заголовки
)

# Инициализация шаблонов и статических файлов
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/voice", StaticFiles(directory="voice"), name="voice")

# Словарь для хранения истории чата каждого пользователя
chat_history_by_user: Dict[str, List[str]] = {}

# Логируем успешную инициализацию переменных и настроек
logging.info("Успешно инициализированы переменные и настройки.")

# Функция для загрузки данных из файла
def load_data(file_name: str) -> dict:
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    return data


# Чтение данных из обоих файлов
admin_data = load_data('admin_questions.json')
user_data = load_data('users_questions.json')

# Объединение данных
data_combined = admin_data + user_data

# Получение списка вопросов из обоих файлов
questions = [item['question'] for item in data_combined]


# Векторизация вопросов
tfidf_vectorizer = TfidfVectorizer()
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    logging.info("Успешно выполнено векторизация admin_user_questions!>")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации admin_user_questions!>: {e}")


# Чтение данных контекстов
contexts_data = load_data('contexts.json')
contexts = [item['context'] for item in contexts_data]


# Векторизация контекстов
contexts_vectorizer = TfidfVectorizer()
try:
    contexts_vectors = contexts_vectorizer.fit_transform(contexts)
    logging.info("Успешно выполнено векторизация текстов!>")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации текстов!>: {e}")


# Функции для сохранения данных и добавления записей в словарь
def save_data(data: dict, file_name: str) -> None:
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def add_entry(dictionary: dict, key: str, value) -> None:
    if key in dictionary:
        if value not in dictionary[key]:
            dictionary[key].append(value)
    else:
        dictionary[key] = [value]


# Загрузка истории чата из файла
file_name = "data.json"
data = load_data(file_name)


# Функция для синтеза речи из текста
async def process_chunk(voice: str, text_chunk: str, output_file: str) -> None:
    # Инициализация объекта для синтеза речи и сохранение файла
    communicate = edge_tts.Communicate(text_chunk, voice)
    await communicate.save(output_file)

# Функция для автоматического выбора голоса в зависимости от языка
def get_voice_for_language(language: str) -> str:
    # Сопоставление языков с голосами
    voices = {
        'en': 'en-IE-EmilyNeural',    # Английский (Ирландия)
        'ru': 'ru-RU-SvetlanaNeural', # Русский (Россия)
        'es': 'es-US-PalomaNeural',   # Испанский (США)
        'fr': 'fr-FR-DeniseNeural',   # Французский (Франция)
        'de': 'de-DE-KatjaNeural',    # Немецкий (Германия)
        'it': 'it-IT-ElsaNeural',     # Итальянский (Италия)
        'zh': 'zh-CN-XiaoxiaoNeural', # Китайский (упрощенный)
        'ja': 'ja-JP-NanamiNeural',   # Японский (Япония)
        'ko': 'ko-KR-SunHiNeural',    # Корейский (Южная Корея)
        'ar': 'ar-AE-FatimaNeural',   # Арабский (Саудовская Аравия)
        'tr': 'tr-TR-EmelNeural',     # Турецкий (Турция)
        'id': 'id-ID-GadisNeural',    # Индонезийский (Индонезия)
        'hi': 'hi-IN-SwaraNeural',    # Хинди (Индия)
        'ur': 'ur-IN-GulNeural',      # Урду (Пакистан)
        'pl': 'pl-PL-ZofiaNeural',    # Польский (Польша)
        'nl': 'nl-BE-DenaNeural',     # Нидерландский (Нидерланды)
        'sv': 'sv-SE-SofieNeural',    # Шведский (Швеция)
        'da': 'da-DK-ChristelNeural', # Датский (Дания)
        'fi': 'fi-FI-NooraNeural'     # Финский (Финляндия)
    }

    # Возвращаем голос для соответствующего языка, если найден, или по умолчанию для английского
    return voices.get(language[:2], 'ru-RU-SvetlanaNeural')

# Функция для синтеза всего текста
async def synthesis(data: str, prefix: str = synthesis_path) -> Tuple[List[str], str]:
    # Определяем язык текста
    detected_language = detect(data)
    
    # Получаем голос для синтеза в зависимости от языка
    voice = get_voice_for_language(detected_language)
    
    if detected_language == "ru":
        data = re.sub(r'!\s*', '. ', data)

    unique_id = uuid.uuid4()  # Генерируем уникальный идентификатор для файла
    created_files = []  # Список для хранения путей к созданным файлам

    # Формируем путь к выходному файлу с уникальным именем
    output_file = os.path.join(prefix, f"synthesis_{unique_id}.mp3")  # Уникальный путь к файлу
    await process_chunk(voice, data, output_file)  # Синтезируем текст в файл

    # Добавляем путь к созданному файлу в список
    created_files.append(output_file)

    # Возвращаем список созданных файлов и уникальный идентификатор
    return created_files, unique_id


async def vectorize_and_find_similarity(question: str, 
                                        tfidf_vectorizer: TfidfVectorizer, 
                                        tfidf_matrix: sp.sparse.csr_matrix) -> Tuple[int, float]:
    if not question.strip():  # Проверяем, что вопрос не пустой
        raise ValueError("Question cannot be empty or just spaces.")
    
    # Преобразуем вопрос в вектор
    question_vector = tfidf_vectorizer.transform([question])

    # Вычисляем косинусное сходство между вопросом и всеми предыдущими вопросами
    cosine_similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    # Находим индекс наиболее похожего вопроса и его степень сходства
    most_similar_index = cosine_similarities.argmax()
    similarity_score = cosine_similarities[most_similar_index]

    return most_similar_index, similarity_score


def vectorize_questions(questions: List[str]) -> Tuple[TfidfVectorizer, sp.sparse.csr_matrix]:
    """
    Функция для векторизации списка вопросов с помощью TF-IDF.
    
    :param questions: Список строк, содержащих вопросы.
    :return: Кортеж с объектом TfidfVectorizer и полученной матрицей TF-IDF.
    :raises ValueError: Если список вопросов пуст.
    """
    if not questions:
        raise ValueError("Список вопросов не может быть пустым.")
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    return tfidf_vectorizer, tfidf_matrix


async def remove_files(files: List[str], prefix: str = synthesis_path) -> None:
    for file in files:
        file_path = os.path.join(prefix, file)

        # Проверяем, существует ли файл
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Файл {file_path} удален успешно.")
            except PermissionError:
                logging.error(f"Отказано в доступе при удалении файла {file_path}. Проверьте права доступа.")
            except OSError as e:
                logging.error(f"Ошибка операционной системы при удалении файла {file_path}: {e}")
            except Exception as e:
                logging.error(f"Неизвестная ошибка при удалении файла {file_path}: {e}")
        else:
            logging.warning(f"Файл {file_path} не существует, не удалось удалить.")


async def append_question_history_to_file(file_name: str = "question_history.txt") -> None:
    try:
        async with aiofiles.open(file_name, "a", encoding="utf-8") as file:
            # Проверяем, есть ли вопросы в истории, чтобы не записывать пустоту
            if question_history:
                await file.write("\n".join(question_history) + "\n")
                logging.info(f"История вопросов успешно добавлена в файл {file_name}.")
            else:
                logging.warning("История вопросов пуста. Нечего записывать.")
    except IOError as e:
        logging.error(f"Ошибка при записи в файл {file_name}: {e}")


@app.get("/download-question-history", tags=["download"])
async def download_txt(response: Response):
    file_path = "question_history.txt"

    if os.path.exists(file_path):
        return FileResponse(path=file_path,
                            filename="question_history.txt",
                            media_type='text/plain')
    else:
        response.status_code = 404
        return {"error": "File not found", "details": "The file 'question_history.txt' does not exist on the server."}

@app.get("/download-bad-answers", tags=["download"])
async def download_bad_answers_txt(response: Response):
    file_path = "bad_answers.txt"

    if os.path.exists(file_path):
        return FileResponse(path=file_path,
                            filename="bad_answers.txt",
                            media_type='text/plain')
    else:
        response.status_code = 404
        return {"error": "File not found", "details": "The file 'bad_answers.txt' does not exist on the server."}

@app.get("/download-json", tags=["download"])
async def download_json(response: Response):
    file_path = "data.json"
    
    if os.path.exists(file_path):
        return FileResponse(path=file_path,
                            filename="data.json",
                            media_type='application/json')
    else:
        response.status_code = 404
        return {"error": "File not found", "details": "The file 'data.json' does not exist on the server."}


def fix_code_block(markdown_text):
    # Регулярное выражение для поиска всех блоков кода с указанием языка
    # Оно ищет такие строки как ```<язык> (например, ```python) и заканчивается на ```
    pattern = r"```(\w+)\n(.*?)```"  # Пример: ```python ... код ... ```

    # Функция для замены блока кода
    def replace_language(match):
        language = match.group(1)  # Извлекаем язык (например, python)
        code = match.group(2)  # Извлекаем сам код
        return f"```{language}\n{code}\n```"  # Возвращаем исправленный блок кода
    
    # Заменяем все вхождения
    fixed_text = re.sub(pattern, replace_language, markdown_text, flags=re.DOTALL)
    return fixed_text


class GitHubRenderer(mistune.HTMLRenderer):
    def block_code(self, text, info=None):
        language = info or "text"  # Если язык не указан, используем "text"
        try:
            # Используем stripall=True для удаления лишних пробелов
            lexer = get_lexer_by_name(language, stripall=True)
        except Exception:
            # Если лексер для языка не найден, используем текстовый лексер
            lexer = get_lexer_by_name("text", stripall=True)

        # Используем GitHub стиль оформления
        formatter = HtmlFormatter(style="github-dark")  # Можно сменить на "github-light"
        return highlight(text, lexer, formatter)


def get_chat_history(user_id: str, max_messages: Optional[int] = None) -> List[str]:
    """
    Функция для получения истории чатов пользователя с возможностью ограничения по количеству сообщений.

    :param user_id: Идентификатор пользователя, для которого нужно получить историю чатов.
    :param max_messages: Максимальное количество сообщений, которое нужно вернуть. Если не указано, возвращается вся история.
    :return: Список сообщений чата пользователя.
    """
    # Получаем историю сообщений пользователя, если она существует, иначе — пустой список
    history = chat_history_by_user.get(user_id, [])

    # Если max_messages задан, ограничиваем размер истории
    if max_messages is not None:
        return history[-max_messages:]
    
    # Возвращаем всю историю, если max_messages не задан
    return history


@app.get("/get_response", tags=["main"])
async def get_response(user_id: str, question: str):
    question = question.strip()
    question_history.append(question)

    # Получаем историю чатов пользователя
    chat_history = get_chat_history(user_id)

    # print('история:')
    # print(chat_history)

    # Если размер истории вопросов достиг максимального значения, записываем в файл и очищаем
    if len(question_history) >= MAX_QUESTION_HISTORY_SIZE:
        await append_question_history_to_file()  # Записываем в файл
        question_history.clear()  # Очищаем историю вопросов

    # Логика для обработки запроса с использованием TF-IDF и нахождения наиболее подходящего ответа
    most_similar_index, accuracy = await vectorize_and_find_similarity(question, tfidf_vectorizer, tfidf_matrix)
    # print(f"Точность: {accuracy}")

    if accuracy >= 0.9:  # Если точность ответа >= 0.9, выбираем наиболее похожий ответ
        most_similar_answer = data_combined[most_similar_index]['answer']
        print('Ответ найден с высокой точностью')

        # Добавляем вопрос и ответ в историю чатов
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": most_similar_answer})

        # Обновляем историю чатов для этого пользователя
        chat_history_by_user[user_id] = chat_history

        # Добавляем запись в данные
        add_entry(data, user_id, chat_history)

        # Сохранение данных в файл
        save_data(data, file_name)

        return {
            "question": question,
            "response": most_similar_answer,
        }
    else:
        print('Ответ с низкой точностью, продолжаем поиск в контексте')

        # Если точность ниже 0.9, пытаемся найти более подходящий ответ через контекст
        # messages = chat_history + [{"role": "user", "content": question}]  # Заменить на template если без кастомной личности

        # Ищем наиболее подходящий контекст
        similarities = cosine_similarity(contexts_vectors, contexts_vectorizer.transform([question]))
        best_index = similarities.argmax()
        if similarities[best_index] < 0.3:  # Пороговое значение схожести
            context = None  # Или другой способ обработки
        else:
            context = contexts[best_index]

        template = f"""
        Ты — виртуальная помощница, всегда готовая помочь! Отвечай дружелюбно, но по делу. Общайся как девушка, без лишних формальностей.  
        Полагайся на историю сообщений, чтобы поддерживать контекст. Отвечай на языке вопроса.

        История сообщений: {chat_history}  
        Контекст: {context}  
        Вопрос: {question}  

        Ответ:
        """

        # print(f"Контекст: {context}")

        try:
            # Запрос к модели для генерации ответа
            response = await g4f.ChatCompletion.create_async(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": template}],
                provider=g4f.Provider.DDG
            )

            # Обработка ответа, исправление блоков кода
            fixed_text = fix_code_block(response)

            # Используем mistune для преобразования в HTML
            markdown = mistune.create_markdown(renderer=GitHubRenderer())
            html = markdown(fixed_text)

            # Добавляем вопрос и ответ в историю чатов
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": html})

            # Обновляем историю чатов для пользователя
            chat_history_by_user[user_id] = chat_history

            # Добавляем запись в данные
            add_entry(data, user_id, chat_history)

            # Сохранение данных в файл
            save_data(data, file_name)
            
        except Exception as e:
            # Обработка ошибок
            raise HTTPException(status_code=500, detail=f"Ошибка при генерации ответа: {str(e)}")
        
        return {"question": question, "response": html}
    

# Эндпоинт для добавления нового вопроса
@app.get("/add_question", tags=["questions"])
async def add_question(question: str, answer: str, added_by: str = "user", password: str = None):
    global tfidf_vectorizer, tfidf_matrix  # Указываем, что обновляем глобальные переменные

    # Проверка пароля для администратора
    if added_by == "admin":
        if password != ADMIN_PASSWORD:
            raise HTTPException(status_code=403, detail="Неверный пароль для администратора.")
        
    question = question.strip()

    # Проверяем, есть ли уже такой вопрос в базе
    if any(item['question'] == question for item in data_combined):
        raise HTTPException(status_code=400, detail="Этот вопрос уже существует в базе.")

    # Генерация нового id для вопроса
    new_id = str(uuid.uuid4())

    # Добавляем новый вопрос в базу данных с указанием, кто добавил
    new_question = {
        "question": question,
        "answer": answer,
        "id": new_id,
        "added_by": added_by  # Указываем, кто добавил вопрос
    }

    # В зависимости от того, кто добавил вопрос, сохраняем его в соответствующий файл
    if added_by == "admin":
        admin_data.append(new_question)
        save_data(admin_data, 'admin_questions.json')
    else:
        user_data.append(new_question)
        save_data(user_data, 'users_questions.json')
    
    # Обновляем комбинированные данные и векторизацию
    data_combined.append(new_question)
    questions.append(question)

    # Пересчитываем векторизацию для обновленных данных
    tfidf_vectorizer, tfidf_matrix = vectorize_questions(questions)

    return {
        "message": "Вопрос успешно добавлен!",
        "question": question,
        "answer": answer,
        "added_by": added_by  # Возвращаем, кто добавил
    }

# Эндпоинт для удаления вопроса
@app.get("/delete_question", tags=["questions"])
async def delete_question(question: str, added_by: str = "admin", password: str = None):
    global data_combined, admin_data, user_data, questions, tfidf_vectorizer, tfidf_matrix

    # Ищем вопрос по тексту
    question_found = next((item for item in data_combined if item['question'] == question), None)

    if not question_found:
        raise HTTPException(status_code=404, detail="Вопрос не найден.")

    # Проверка пароля для админа
    if added_by == "admin" and password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Неверный пароль для администратора.")

    # Если вопрос был добавлен пользователем, проверяем, что он удаляет только свой вопрос
    if added_by == "user" and question_found['added_by'] != "user":
        raise HTTPException(status_code=403, detail="Пользователь не может удалять вопросы, добавленные администратором.")

    # Удаляем вопрос из соответствующего списка (admin_data или user_data)
    if question_found in admin_data:
        admin_data.remove(question_found)
        save_data(admin_data, 'admin_questions.json')
    elif question_found in user_data:
        user_data.remove(question_found)
        save_data(user_data, 'users_questions.json')

    # Удаляем вопрос из комбинированных данных
    data_combined = [item for item in data_combined if item != question_found]

    # Обновляем список вопросов
    questions = [item['question'] for item in data_combined]

    # Пересчитываем векторизацию
    tfidf_vectorizer, tfidf_matrix = vectorize_questions(questions)

    return {"message": "Вопрос успешно удален!"}

# Эндпоинт для получения вопросов, добавленных пользователями или администраторами
@app.get("/get_questions_by_source", tags=["questions"])
async def get_questions_by_source(source: str = "all"):
    """
    Получение вопросов по источнику: 'all' (все вопросы), 'user' (вопросы, добавленные пользователем),
    'admin' (вопросы, добавленные администратором).
    
    :param source: Источник вопросов ('all', 'user', 'admin'). По умолчанию 'all'.
    :return: Список вопросов по указанному источнику.
    :raises HTTPException: Если передано неверное значение для параметра source.
    """
    # Проверка корректности параметра source
    if source not in ["admin", "user", "all"]:
        raise HTTPException(status_code=400, detail="Неверное значение для параметра source. Ожидается 'admin', 'user', или 'all'.")
    
    # Возвращаем все вопросы, если источник 'all'
    if source == "all":
        return data_combined
    
    # Фильтруем и возвращаем вопросы по источнику ('admin' или 'user')
    return [item for item in data_combined if item["added_by"] == source]
    

@app.get("/add_context", tags=["contexsts"])
async def add_context(new_context: str, added_by: str = "admin", password: str = None):
    global contexts, contexts_vectorizer, contexts_vectors  # Указываем, что обновляем глобальные переменные

    # Проверка доступа для администратора
    if added_by != "admin" or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Доступ запрещен.")
    
    # Убираем пробелы в начале и конце контекста
    new_context = new_context.strip()

    # Проверка на пустое значение контекста
    if not new_context:
        raise HTTPException(status_code=400, detail="Контекст не может быть пустым.")
    
    # Проверяем, существует ли уже такой контекст
    if any(item['context'] == new_context for item in contexts_data):
        raise HTTPException(status_code=400, detail="Этот контекст уже существует в базе.")
        
    # Генерация нового id для вопроса
    new_id = str(uuid.uuid4())

    # Добавляем новый контекст в базу данных
    new_entry = {
        "context": new_context,
        "id": new_id,
        "added_by": added_by
    }

    # Добавляем в данные и сохраняем
    contexts_data.append(new_entry)
    save_data(contexts_data, 'contexts.json')
    
    # Обновляем список контекстов из данных
    contexts = [item["context"] for item in contexts_data]
    
    # Пересчитываем векторизацию
    contexts_vectorizer, contexts_vectors = vectorize_questions(contexts)
    
    return {"message": "Контекст успешно добавлен!"}

# Эндпоинт для удаления контекста
@app.get("/delete_context", tags=["contexsts"])
async def delete_context(context: str, added_by: str = "admin", password: str = None):
    global contexts_data, contexts, contexts_vectorizer, contexts_vectors

    # Ищем контекст по тексту
    context_found = next((item for item in contexts_data if item['context'] == context), None)

    if not context_found:
        raise HTTPException(status_code=404, detail="Контекст не найден.")

    # Проверка прав доступа для администратора
    if added_by == "admin" and password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Неверный пароль для администратора.")

    # Проверка, что пользователь может удалять только свои контексты
    if added_by == "user" and context_found['added_by'] != "user":
        raise HTTPException(status_code=403, detail="Пользователь не может удалять контексты, добавленные Админом.")

    # Удаляем контекст из данных и сохраняем изменения
    contexts_data.remove(context_found)
    save_data(contexts_data, 'contexts.json')

    # Обновляем список контекстов
    contexts = [item['context'] for item in contexts_data]

    # Переобучаем векторизацию
    contexts_vectorizer, contexts_vectors = vectorize_questions(contexts)

    return {"message": "Контекст успешно удален!"}

# Эндпоинт для получения контекстов, добавленных пользователями или администраторами
@app.get("/get_contexts_by_source", tags=["contexsts"])
async def get_contexts_by_source(source: str = "all"):
    """
    Получение контекстов по источнику: 'all' (все контексты), 'user' (контексты, добавленные пользователем),
    'admin' (контексты, добавленные администратором).
    
    :param source: Источник контекстов ('all', 'user', 'admin'). По умолчанию 'all'.
    :return: Список контекстов по указанному источнику.
    :raises HTTPException: Если передано неверное значение для параметра source.
    """
    # Проверяем корректность значения параметра source
    if source not in ["admin", "user", "all"]:
        raise HTTPException(status_code=400, detail="Неверное значение для параметра source.")
    
    # Возвращаем все контексты, если источник "all"
    if source == "all":
        return contexts_data
    
    # Фильтруем контексты по добавившему пользователю или админу
    return [item for item in contexts_data if item["added_by"] == source]


@app.get("/", response_class=HTMLResponse, tags=["site"])
async def home(request: Request, response: Response):
    # Получение user_id из cookie или создание 
    user_id = request.cookies.get("user_id") or str(uuid.uuid4())

    await get_info(request, user_id)

    # Получение истории сообщений пользователя
    chat_history = get_chat_history(user_id)

    # Подсчет количества сообщений
    message_count = len(chat_history)
    
    # Ответ с использованием шаблона
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history,
        "message_count": message_count
    })
    
    # Устанавливаем или обновляем cookie с user_id, сроком действия 7 дней
    response.set_cookie("user_id", user_id, max_age=604800, httponly=True)  # Cookie с максимальным сроком 7 дней

    return response

@app.post("/api", tags=["site"]) 
async def get_answer(request: Request, question: str = Form(...)):
    # Получаем user_id из cookie
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Session ID is missing")
    
    # Проверка на пустой вопрос
    if not question.strip():
        return JSONResponse({"error": "Empty message"}, status_code=400)
    
    try:
        # Отправляем запрос на другой эндпоинт для получения ответа
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://helpbotx.onrender.com/get_response",
                params={"user_id": user_id, "question": question},
                timeout=20  # Время ожидания ответа
            )
            # Проверяем, что запрос прошел успешно
            response.raise_for_status()

            # Парсим ответ
            response_data = response.json()

            # Если в ответе есть нужное поле "response", возвращаем его
            if "response" in response_data:
                return JSONResponse({"question": question, "response": response_data["response"]})

            # Если поле "response" отсутствует в ответе
            return JSONResponse({"error": "No response field in the returned data"}, status_code=500)
    except httpx.HTTPStatusError as e:
        # Обработка ошибок HTTP запросов (например, 404, 500 и т.п.)
        return JSONResponse({"error": f"HTTP error: {e.response.status_code}, {e.response.text}"}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        # Обработка ошибок запроса (например, не удалось подключиться)
        return JSONResponse({"error": f"Error with the request: {str(e)}"}, status_code=500)
    except Exception as e:
        # Общая обработка исключений
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)


@app.get("/send_files", tags=["download"])
async def send_files_endpoint():
    file_paths = ['data.json', 'logs.json', 'contexts.json', 'admin_questions.json', 'users_questions.json']
    await send_files(file_paths)
    return {"message": "Файлы отправлены."}


@app.get("/get-info", tags=["site"])
async def get_info(request: Request, user_id: str = None):
    # Извлекаем IP-адрес из заголовка X-Forwarded-For, если доступен
    forwarded_for = request.headers.get('X-Forwarded-For')
    client_ip = forwarded_for.split(',')[0] if forwarded_for else request.client.host

    # Извлекаем User-Agent, если доступен
    user_agent = request.headers.get('User-Agent', 'Unknown')

    # Получаем все заголовки в виде словаря
    headers = dict(request.headers)

    # Формируем информацию для сохранения в файл
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "ip_address": client_ip,
        "user_id": user_id,
        "user_agent": user_agent,
        "headers": headers
    }

    # Сохраняем информацию в файл (например, logs.json) с отступом 4 пробела
    try:
        with open("logs.json", "r+") as log_file:
            # Читаем текущие данные из файла
            try:
                current_data = json.load(log_file)
            except json.JSONDecodeError:
                current_data = []

            # Добавляем новый лог в существующие данные
            current_data.append(log_data)

            # Перемещаемся в начало файла и записываем обновленные данные
            log_file.seek(0)
            json.dump(current_data, log_file, indent=4)
            log_file.truncate()  # Убираем старые данные, если файл был обрезан

    except FileNotFoundError:
        # Если файл не существует, создаем новый и записываем данные
        with open("logs.json", "w") as log_file:
            json.dump([log_data], log_file, indent=4)

    # Возвращаем информацию клиенту
    return {
        "ip_address": client_ip,
        "user_agent": user_agent,
        "headers": headers
    }


@app.get("/synthesis", tags=["main"])
async def process_request(data: str):
    try:
        created_file = await synthesis(data)
        return {"created_file": created_file[0]}
    
    except Exception as e:
        return {"error": str(e)}


class AnswerData(BaseModel):
    user: str
    assistant: str


@app.post("/bad_answer", tags=["site"])
async def bad_answer_process(request: Request, data: AnswerData):
    user_id = request.cookies.get("user_id") or "anonymous"

    async with aiofiles.open("bad_answers.txt", "a", encoding="utf-8") as file:
        await file.write(f"User ID: {user_id}\n")
        await file.write(f"User: {data.user}\n")
        await file.write(f"Assistant: {data.assistant}\n")
        await file.write("\n")
    return {"message": "Данные успешно получены"}


@app.get("/clear_history", tags=["site"])
async def clear_history(request: Request) -> RedirectResponse:
    # Получаем session_id из cookies
    session_id = request.cookies.get("user_id")

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is missing")

    # Считываем файлы из директории и удаляем их
    file_list = os.listdir(synthesis_path)

    # Удаляем файлы
    await remove_files(files=file_list)

    # Очищаем историю чатов для данного пользователя
    chat_history_by_user[session_id] = []

    # Перенаправляем на главную страницу
    return RedirectResponse("/", status_code=303)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def left_chat(self, client_id: str):
        formatted_message = f"""<div class="system-message-content">#{client_id} вышел из чата.</div>"""
        await self._broadcast_to_all(formatted_message)

    async def broadcast(self, message: str, sender: str):
        formatted_message = f"""
            <span class="avatar">
                <img src="static/logo_b.png" alt="Avatar">
            </span>
            <div class="info-user">
                <div class="name-user">{sender}</div>
                <div class="message-content">{message}</div>
            </div>
        """
        await self._broadcast_to_all(formatted_message)

    def get_active_connections_count(self) -> int:
        return len(self.active_connections)
    
    async def _broadcast_to_all(self, message: str):
        """
        Helper method to send a message to all active WebSocket connections.
        
        :param message: The message to be sent to all connections.
        """
        for connection in self.active_connections.values():
            try:
                await connection.send_text(message)
            except Exception as e:
                # Handle potential errors, like the connection being closed
                print(f"Error sending message to a connection: {e}")
                continue


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()

            await manager.broadcast(data, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.left_chat(client_id)

@app.get("/chat", response_class=HTMLResponse, tags=["site"])
async def get(request: Request):
    session_id = request.cookies.get("user_id") or "anonymous"

    # Отправляем страницу chat.html с переданным session_id
    return templates.TemplateResponse("chat.html", {"request": request, "session_id": session_id})


@app.get("/active_connections_count", tags=["site"])
async def get_active_connections_count():
    return {"active_connections_count": manager.get_active_connections_count()}


@app.get("/info", response_class=HTMLResponse, tags=["site"])
async def read_another_page(request: Request):
    return templates.TemplateResponse("info.html", {"request": request})


def main():
    print("Начало запуска...")


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="127.0.0.1", port=8000)