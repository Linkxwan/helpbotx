// Функция, которая фокусирует на поле ввода после загрузки страницы
window.onload = function() {
    document.getElementById("question-input").focus();
};

window.onload = function() {
    window.scrollTo(0, document.body.scrollHeight);
};
function validateForm() {
    var questionInput = document.getElementById("question-input").value;
    if (questionInput.trim() === "") {
        return false;
    }
    return true;
}
// function replaceProgrammingLanguages(text) {
//     const languageReplacements = {
//         'python': 'пайтон',
//         'javascript': 'джавскрипт',
//         'java': 'джава',
//         'ruby': 'руби',
//         'C++': 'си плюс плюс',  // Обработали C++ с учетом символа "+"
//         'c#': 'си шарп',
//         'swift': 'свифт',
//         'php': 'пи эйч пи',
//         'html': 'эйч ти эм эль',
//         'css': 'си эс эс',
//         'typescript': 'тайпскрипт',
//         'go': 'го',
//         'rust': 'раст',
//         'kotlin': 'котлин',
//         'scala': 'скала',
//         'r': 'ар',
//         'sql': 'эс кью эл',
//         'objective-c': 'обджектив си',
//         'matlab': 'матлаб',
//         'vhdl': 'виэйч ди эл',
//         'lua': 'луа',
//         'elixir': 'эликсир',
//         'haskell': 'хаскел',
//         'perl': 'пёрл',
//         'f#': 'эф шарп',
//         'dart': 'дарт',
//         'assembly': 'ассемблер',
//         'delphi': 'дельфи',
//         'actionscript': 'экшнскрипт',
//         'visual basic': 'визуал бейсик',
//         'fortran': 'фортран'
//     };

//     // Пройдем по всем ключам и заменим их на подходящие произношения
//     for (let language in languageReplacements) {
//         // Создаем регулярное выражение для замены, игнорируя регистр
//         const regex = new RegExp(`\\b${language.replace(/[.*+?^=!:${}()|\[\]\/\\]/g, "\\$&")}\\b`, 'gi');
//         text = text.replace(regex, languageReplacements[language]);
//     }

//     return text;
// }


const chatContainer = document.getElementById("chat-container");
    // Функция добавления сообщения
    function addMessage(role, content, isThinking = false) {
    const messages = chatContainer.querySelectorAll(".chat-message-user, .chat-message-bot");
    messages.forEach(msg => msg.querySelector(".info")?.classList.remove("last")); // Убираем старый last

    // Прокручиваем страницу вниз
    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth',
        block: 'end'
    });

    // Получаем элемент presentation внутри обработчика события, чтобы он был доступен
    const presentation = document.querySelector('.presentation');
    const hello = document.getElementById('hello');

    // Если presentation не найден, это может вызвать ошибку
    if (!presentation) {
        console.error("Элемент presentation не найден.");
        return; // Прерываем выполнение, если элемент не найден
    }

    const messageIndex = messages.length; // Новый индекс
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(role === "user" ? "chat-message-user" : "chat-message-bot");

    if (role === "user") {
        messageDiv.innerHTML = `
            <div class="info-user">
                <div class="message-content for-user" id="user-message-${messageIndex}">${content}</div>
            </div>`;
    } else {
        messageDiv.innerHTML = `
            <span class="avatar">
                <img src="static\\logo_b.png" alt="Avatar">
            </span>
            <div class="info-bot">
                
                <div class="message-content" id="assistant-message-${messageIndex}">
                    <div class="loader">
                        <span class="loader__element"></span>
                        <span class="loader__element"></span>
                        <span class="loader__element"></span>
                    </div>
                </div>
                <div class="info last" id="message-info-${messageIndex}">
                    <audio id="myAudio" style="display: none;" controls>
                        <source id="audioSource" type="audio/mpeg">
                        Ваш браузер не поддерживает элемент audio.
                    </audio>
                    <div class="wrap-svg last">
                        <svg fill="#d1d5db" width="20px" height="20px" viewBox="0 0 24 24" id="sound" data-name="Line Color" xmlns="http://www.w3.org/2000/svg" class="icon line-color" stroke="#d1d5db" onclick="sendSynthesisRequest('${messageIndex}')">
                            <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                            <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g>
                            <g id="SVGRepo_iconCarrier">
                                <path id="secondary" d="M18.36,5.64a9,9,0,0,1,0,12.72" style="fill: none; stroke: #d1d5db; stroke-linecap: round; stroke-linejoin: round; stroke-width: 2;"></path>
                                <path id="secondary-2" data-name="secondary" d="M15.54,8.46a5,5,0,0,1,0,7.08" style="fill: none; stroke: #d1d5db; stroke-linecap: round; stroke-linejoin: round; stroke-width: 2;"></path>
                                <path id="primary" d="M11,5V19L7,15H4a1,1,0,0,1-1-1V10A1,1,0,0,1,4,9H7Z" style="fill: none; stroke: #d1d5db; stroke-linecap: round; stroke-linejoin: round; stroke-width: 2;"></path>
                            </g>
                        </svg>
                        <svg fill="#d1d5db" height="20px" width="20px" version="1.1" id="copy-button" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="-46 -46 552.00 552.00" xml:space="preserve" stroke="#d1d5db" stroke-width="0.0046" transform="rotate(0)" onclick="copyText('${messageIndex}')">
                            <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                            <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round" stroke="#CCCCCC" stroke-width="0.9200000000000002"></g>
                            <g id="SVGRepo_iconCarrier"> <g> <g> <g>
                                <path d="M425.934,0H171.662c-18.122,0-32.864,14.743-32.864,32.864v77.134h30V32.864c0-1.579,1.285-2.864,2.864-2.864h254.272 c1.579,0,2.864,1.285,2.864,2.864v254.272c0,1.58-1.285,2.865-2.864,2.865h-74.729v30h74.729 c18.121,0,32.864-14.743,32.864-32.865V32.864C458.797,14.743,444.055,0,425.934,0z"></path>
                                <path d="M288.339,139.998H34.068c-18.122,0-32.865,14.743-32.865,32.865v254.272C1.204,445.257,15.946,460,34.068,460h254.272 c18.122,0,32.865-14.743,32.865-32.864V172.863C321.206,154.741,306.461,139.998,288.339,139.998z M288.341,430H34.068 c-1.58,0-2.865-1.285-2.865-2.864V172.863c0-1.58,1.285-2.865,2.865-2.865h254.272c1.58,0,2.865,1.285,2.865,2.865v254.273h0.001 C291.206,428.715,289.92,430,288.341,430z"></path></g> </g> </g>
                            </g>
                        </svg>
                        <svg fill="#d1d5db" width="20px" height="20px" viewBox="0 0 1024.00 1024.00" id="bad_answer" xmlns="http://www.w3.org/2000/svg" class="icon" transform="rotate(0)matrix(-1, 0, 0, 1, 0, 0)" onclick="bad_answer('${messageIndex}')">
                            <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                            <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g>
                            <g id="SVGRepo_iconCarrier">
                                <path d="M885.9 490.3c3.6-12 5.4-24.4 5.4-37 0-28.3-9.3-55.5-26.1-77.7 3.6-12 5.4-24.4 5.4-37 0-28.3-9.3-55.5-26.1-77.7 3.6-12 5.4-24.4 5.4-37 0-51.6-30.7-98.1-78.3-118.4a66.1 66.1 0 0 0-26.5-5.4H144c-17.7 0-32 14.3-32 32v364c0 17.7 14.3 32 32 32h129.3l85.8 310.8C372.9 889 418.9 924 470.9 924c29.7 0 57.4-11.8 77.9-33.4 20.5-21.5 31-49.7 29.5-79.4l-6-122.9h239.9c12.1 0 23.9-3.2 34.3-9.3 40.4-23.5 65.5-66.1 65.5-111 0-28.3-9.3-55.5-26.1-77.7zM184 456V172h81v284h-81zm627.2 160.4H496.8l9.6 198.4c.6 11.9-4.7 23.1-14.6 30.5-6.1 4.5-13.6 6.8-21.1 6.7a44.28 44.28 0 0 1-42.2-32.3L329 459.2V172h415.4a56.85 56.85 0 0 1 33.6 51.8c0 9.7-2.3 18.9-6.9 27.3l-13.9 25.4 21.9 19a56.76 56.76 0 0 1 19.6 43c0 9.7-2.3 18.9-6.9 27.3l-13.9 25.4 21.9 19a56.76 56.76 0 0 1 19.6 43c0 9.7-2.3 18.9-6.9 27.3l-14 25.5 21.9 19a56.76 56.76 0 0 1 19.6 43c0 19.1-11 37.5-28.8 48.4z"></path>
                            </g>
                        </svg>
                    </div>
                </div>
            </div>`;
    }
    // Проверяем, есть ли сообщения
    if (messageIndex.length > 0) {
            presentation.classList.remove('hidden');
        } else {
            presentation.classList.add('hidden');
            hello.style.display = 'none';
        }


    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Прокрутка вниз
    return messageIndex;
}



let isRequestInProgress = false;  // Флаг для отслеживания состояния запроса

async function sendQuestion(question) {
    try {
        // Если запрос уже в процессе, ничего не делаем
        if (isRequestInProgress) return;

        // Убираем класс "last" у всех предыдущих элементов
        const wrapSvgElements = document.querySelectorAll(".wrap-svg");
        wrapSvgElements.forEach((el) => {
            el.classList.remove("last");
        });

        // Добавляем сообщение пользователя
        addMessage("user", question);

        // Показываем анимацию загрузки
        let botMessageIndex = addMessage("bot", "", true);

        // Устанавливаем флаг в true, чтобы заблокировать кнопку
        isRequestInProgress = true;
        document.getElementById("send-button").disabled = true; // Блокируем кнопку отправки

        // Создаем FormData и отправляем запрос
        const formData = new FormData();
        formData.append('question', question);

        const response = await fetch('/api', {
            method: 'POST',
            body: formData
        });

        // Проверяем успешность запроса
        if (!response.ok) {
            throw new Error('Ошибка сети');
        }

        const data = await response.json();

        if (data.response) {
            // Заменяем анимацию на текст ответа
            let botMessageElement = document.getElementById(`assistant-message-${botMessageIndex}`);
            botMessageElement.innerHTML = data.response;

            // Обновляем анимацию (если требуется)
            updateHighlightWidth();
        } else {
            let botMessageElement = document.getElementById(`assistant-message-${botMessageIndex}`);
            botMessageElement.textContent = `Ошибка: ${data.error}`;
        }

        // Добавляем класс "last" к последнему элементу
        const updatedWrapSvgElements = document.querySelectorAll(".wrap-svg");
        const lastWrapSvgElement = updatedWrapSvgElements[updatedWrapSvgElements.length - 1];
        lastWrapSvgElement.classList.add("last");

        // Прокручиваем страницу вниз
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: "smooth"
        });

    } catch (error) {
        console.error('Произошла ошибка:', error);
    } finally {
        // Разблокируем кнопку и сбрасываем флаг после завершения запроса
        isRequestInProgress = false;
        document.getElementById("send-button").disabled = false;
    }
}




async function sendSynthesisRequest(index) {
    const messageElement = document.getElementById(`assistant-message-${index}`);
    
    // Создаем копию элемента, чтобы не изменять оригинальную структуру
    const cloneElement = messageElement.cloneNode(true);

    // Удаляем все теги <pre> из клонированного элемента
    const preElements = cloneElement.getElementsByTagName('pre');
    Array.from(preElements).forEach(pre => pre.remove());

    // Получаем текст из клонированного элемента без <pre>
    let messageContent = cloneElement.textContent.trim();
    
    // Заменяем все восклицательные знаки на точки
    // messageContent = messageContent.replace(/!/g, ',');
    
    // Удаляем эмодзи
    messageContent = messageContent.replace(/[\p{Emoji}]/gu, '');

    // messageContent = replaceProgrammingLanguages(messageContent);
    console.log('Текст после первичной обработки:', messageContent);

    // Отправляем запрос с очищенным текстом
    const response = await fetch(`/synthesis?data=${encodeURIComponent(messageContent)}`, {
        method: 'GET',
    });

    if (!response.ok) {
        throw new Error('Ошибка отправки запроса');
    }

    const responseData = await response.json(); // или response.text() в зависимости от формата ответа
    const audioURL = responseData['created_file'];
    console.log(audioURL); // вывод данных в консоль
    playAudio(audioURL);
}



function playAudio(audioURL) {
    const audio = new Audio(audioURL);
    audio.play();
}

function copyText(index) {
    const messageContent = document.getElementById(`assistant-message-${index}`).textContent;

    navigator.clipboard.writeText(messageContent)
        .then(() => {
            const notification = document.getElementById('copy-notification');
            notification.style.display = 'block';
            setTimeout(() => {
                notification.style.display = 'none';
            }, 2000); // Скрыть уведомление через 2 секунды
        });
}

function bad_answer(index) {
    const userMessage = document.getElementById(`user-message-${index - 1}`).textContent;
    const assistantMessage = document.getElementById(`assistant-message-${index}`).textContent;

    const data = {
        user: userMessage,
        assistant: assistantMessage
    };

    fetch('/bad_answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Ошибка отправки запроса');
        }
        console.log('Данные успешно отправлены');
    })
    .catch(error => {
        console.error(error);
    });
}

document.getElementById("clear-history").addEventListener("click", function() {
    fetch("/clear_history")
        .then(response => {
            if (response.ok) {
                window.location.reload();
            } else {
                console.error("Ошибка при очистке истории чата");
            }
        });
});
// // Функция для добавления эффекта появления текста по буквам
// function typeMessage(element) {
//     var message = element.textContent;
//     element.textContent = ''; // Очищаем содержимое элемента
//     var index = 0;
//     var typingInterval = setInterval(function() {
//         if (index < message.length) {
//             element.textContent += message.charAt(index);
//             index++;
//         } else {
//             clearInterval(typingInterval);
//         }
//     }, 10); // Задержка между появлением букв (в миллисекундах)
// }
// // Получаем последний элемент с классом "message-content" и вызываем функцию для добавления эффекта появления по буквам
// var messageContentElements = document.querySelectorAll('.message-content');
// var lastMessageElement = messageContentElements[messageContentElements.length - 1];
// if (lastMessageElement) {
//     typeMessage(lastMessageElement);
// }


// // ================ если нужно будет убрать имя бота ======================
function removeTopMarginForFirstP() {
    // Получаем все блоки с классом .chat-message-bot
    const botMessages = document.querySelectorAll('.chat-message-bot');

    // Проходим по каждому блоку
    botMessages.forEach((message) => {
        // Находим все элементы <p> внутри текущего блока
        const paragraphs = message.querySelectorAll('p');

        // Если есть хотя бы один элемент <p>, устанавливаем margin-top для первого
        if (paragraphs.length > 0) {
            paragraphs[0].style.marginTop = '0px';
        }
    });
}

// Вызываем функцию, чтобы применить стили
removeTopMarginForFirstP();
// // ========================================================================

// Выбираем textarea
var textarea = document.getElementById("question-input");

textarea.addEventListener("input", function(e) {
  if (e.inputType !== "insertText") return; // Игнорируем события, не связанные с добавлением текста

  // Обновляем высоту textarea
  this.style.height = "auto";
  this.style.height = (this.scrollHeight) + "px";
});

textarea.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && e.shiftKey) {
    var cursorPosition = this.selectionStart; // Определяем позицию курсора
    var textBefore = this.value.substring(0, cursorPosition); // Текст до позиции курсора
    var textAfter = this.value.substring(cursorPosition, this.value.length); // Текст после позиции курсора

    // Объединяем текст до и после курсора с новой строкой между ними
    this.value = textBefore + '\n' + textAfter;

    // Перемещаем курсор на новую строку
    this.selectionStart = this.selectionEnd = cursorPosition + 1;

    this.style.height = "auto"; // Обновляем высоту textarea после добавления новой строки
    this.style.height = (this.scrollHeight) + "px";
    
    e.preventDefault(); // Предотвращаем действие по умолчанию (перенос строки)
  }
});

document.getElementById('question-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault(); // Предотвращаем действие по умолчанию (в данном случае перенос строки)
        document.getElementById('send-button').click(); // Вызываем нажатие на кнопку
    }
});

textarea.addEventListener("input", function() {
  // Устанавливаем высоту textarea на основе содержимого
  this.style.height = "auto";
  this.style.height = (this.scrollHeight) + "px";
});

// Получаем ссылку на textarea
var textarea = document.getElementById('question-input');

// Добавляем обработчик события input, который срабатывает при изменении содержимого textarea
textarea.addEventListener('input', function() {
    var button = document.getElementById('send-button');
    // Если в textarea есть текст, добавляем кнопке класс active
    if (this.value.trim() !== '') {
        button.classList.add('active');
    } else {
        button.classList.remove('active');
    }
});