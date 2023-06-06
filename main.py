#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.
# https://web.telegram.org/k/#@LungHealthbot
"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

from telegram import __version__ as TG_VER
import keras

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler

import os
from dotenv import load_dotenv
from io import BytesIO
import numpy as np
from keras.preprocessing import image
from keras import utils
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from PIL import Image
import requests
from io import BytesIO

load_dotenv()

TOKEN = os.getenv('TOKEN')

model = keras.models.load_model('KerasModel')
print(dir(model))
# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

path = './images/'
test_datagen = ImageDataGenerator(rescale=1./255)

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
"""
Добро пожаловать в телеграмм-бот для анализа рентгеновских снимков для диагностики заболеваний легких!
Мы рады приветствовать вас в этом инновационном сервисе, который поможет вам быстро и точно оценить состояние ваших легких на основе рентгеновских снимков. Наш бот основан на передовых алгоритмах и искусственном интеллекте, чтобы предоставить вам качественные результаты.
Загрузите ваш рентгеновский снимок, и наш бот проведет анализ, выявив возможные патологии и помогая вам лучше понять ваше состояние. Наша система обучена распознавать различные заболевания, включая пневмонию, туберкулез, Ковид-19 и отличать их от снимков здоровых легких. Получите быстрый и надежный диагноз прямо здесь!
Пожалуйста, имейте в виду, что результаты, предоставляемые нашим ботом, не заменяют медицинскую консультацию специалиста. Они являются лишь предварительной оценкой и рекомендуется обратиться к врачу для окончательного диагноза.
Мы надеемся, что наш телеграмм-бот сможет облегчить процесс диагностики и помочь вам заботиться о вашем здоровье. Не стесняйтесь задавать вопросы, если у вас возникнут, и давайте начнем!

Загрузите изображение с камеры или из галереи.
Также вы можете отправить ссылку на изображение. Примеры ссылок:
<a href="https://www.kenhub.com/thumbor/AkXFsw0396y894sLEMWlcDuChJA=/fit-in/800x1600/filters:watermark(/images/logo_url.png,-10,-10,0):background_color(FFFFFF):format(jpeg)/images/library/10851/eXtmE1V2XgsjZK2JolVQ5g_Border_of_left_atrium.png"> Норма </a>
<a href="https://www.wikidoc.org/images/8/85/Pulmonary_Tuberculosis_X-ray3.jpg?20140905151403"> Туберкулез </a>
<a href="https://storage.googleapis.com/kagglesdsdata/datasets/1592399/2619910/test/PNEUMONIA/person100_bacteria_475.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230606%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230606T111645Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=0b063c5274f1d54d73533a65e97a09906784684ebae1f34cf86cb00467ade420093a39e50da191ef1f973501425601657892ba10f0748ef50ccdd2c49910724735043d6ed6083a2a63438c1593ce4bcdaa97b333d203034e7f0c052360c475de0ebe1d7d287eae7d8b947e34130fb613651924eda406fac52a38f444fe4e3641f12bd1de25174ef93da0c03ebb97dfb367c8acf09b9446daace458457d7398145fb2a523471de8fc8cb38f0fc1b552e74db1f3d95e4e1cf20a2dad0ed91d492bfc1dca609cca2b586d21a819e25c60a726f0c2a9a7cb25f8f8f57c05f11dc2da19fd9145345fb31f7f29d9cf289dd1c99e3dc53bf09c7308765e938c838ee276"> Пневмония </a>
<a href="https://tidsskriftet.no/sites/default/files/styles/default_scale/public/article--2020--04--20-0332--KRO_20-0332-01_ENG.jpg?itok=RnC4A_09"> Ковид-19 </a>
"""
    )

async def upload_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    try:
        if message.text:
            response = requests.get(message.text)
            img = Image.open(BytesIO(response.content))
            img.save("./images/test/test.jpg")

    except:
        await update.message.reply_text("Изображение из этого источника не может быть загружено.")

        return
    if message.photo:
        # Телеграмм отправляет фото в различных размерах, мы возьмем самое большое
        photo = message.photo[-1]
        file = await context.bot.getFile(photo.file_id)
        await file.download_to_drive("./images/test/test.jpg")
    test_generator = test_datagen.flow_from_directory(directory=path,
                                                  target_size=(224, 224),
                                                  color_mode="grayscale",
                                                  batch_size=1,
                                                  class_mode=None,
                                                  shuffle=False,
                                                  seed=42)
    #Predict Output
    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
    test_generator.reset()
    pred=model.predict(test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)

    predicted_class_indices=np.argmax(pred,axis=1)
    labels = {0: 'Ковид19', 1: 'Норма', 2: 'Пневмония', 3: 'Туберкулез'}
    predictions = [labels[k] for k in predicted_class_indices]
    res=dict(zip(list(labels.values()),list(pred[0])))
    sorted_tuple=sorted(res.items(),key=lambda x:x[1],reverse=True)
    converted_res = dict(sorted_tuple)
    ans="Вероятность:\n"
    for k,v in converted_res.items():
        per=v*100
        ans+= f"{k} -- {per:.1f}%\n"
    print(converted_res)
    await update.message.reply_text(ans)

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.ALL, upload_image))


    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()