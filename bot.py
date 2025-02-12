import os

TOKEN = os.environ["TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]


# bot.py
from telegram import Bot, InputMediaDocument


# Asynchronous function to send the file
async def send_file(file_path):
    try:
        bot = Bot(token=TOKEN)
        with open(file_path, 'rb') as file:
            await bot.send_document(chat_id=CHAT_ID, document=file)
    except Exception as e:
        print(f"Error occurred: {e}")


# Asynchronous function to send multiple files
async def send_files(file_paths):
    try:
        bot = Bot(token=TOKEN)
        media_group = []

        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                media_group.append(InputMediaDocument(file.read(), filename=file_path.split("/")[-1]))

        await bot.send_media_group(chat_id=CHAT_ID, media=media_group)

    except Exception as e:
        print(f"Error occurred: {e}")
        

# Asynchronous function to send the file
async def server_sturtup():
    try:
        bot = Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text="Тик-так! Сервер запустился! ⏰\nТеперь я готова помочь! 🥳💖")
    except Exception as e:
        print(f"Error occurred: {e}")

# Asynchronous function to send the file
async def server_shutdown():
    try:
        bot = Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text="Сервер остановлен! 💤\nВремя для небольшого отдыха!")
    except Exception as e:
        print(f"Error occurred: {e}")




# import requests


# url = f'https://api.telegram.org/bot{TOKEN}/getUpdates'

# response = requests.get(url)
# updates = response.json()

# # Посмотрим на структуру ответа, чтобы найти chat_id
# print(updates)
