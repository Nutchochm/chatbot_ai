class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            loadingDiv: document.getElementById("loading")
        };

        this.state = false;
        this.message = [];
    }

    // Simulate a chatbot response
    simulateChatbotResponse = (message) => {
        // Show loading animation
        loadingDiv.classList.remove("hidden");

        // Simulate delay for response
        setTimeout(() => {
        loadingDiv.classList.add("hidden");
        appendMessage("Chatbot", "This is a response to: " + message);
        }, 2000); // 2-second delay
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const inputField = chatBox.querySelector('input');
        inputField.addEventListener("keyup", (event) => {
            if (event.key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatBox) {
        this.state = !this.state;
        if (this.state) {
            chatBox.classList.add('chatbox--active');
        } else {
            chatBox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatBox) {
        const textField = chatBox.querySelector('input');
        const text1 = textField.value.trim();
        if (!text1) {
            return;
        }

        const msg1 = { name: "User", message: text1 };
        this.message.push(msg1);

        // Add initial loading message
        const loadingMessage = { name: 'Sam', message: "." };
        this.message.push(loadingMessage);
        this.updateChatText(chatBox);

        // Cycle loading dots
        let dotCount = 0;
        const loadingInterval = setInterval(() => {
            dotCount = (dotCount % 3) + 1; // Cycle between 1, 2, and 3 dots
            loadingMessage.message = ".".repeat(dotCount);
            this.updateChatText(chatBox); // Update the chat to reflect changes
        }, 500); // Update every 500ms

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then((response) => response.json())
            .then((data) => {
                clearInterval(loadingInterval);
                this.message.pop();
                const msg2 = { name: 'Sam', message: data.answer };
                this.message.push(msg2);
                this.updateChatText(chatBox);
                textField.value = ""; // Clear input on success
            })
            .catch((error) => {
                clearInterval(loadingInterval);
                console.error('Error:', error);
                const msg2 = { name: 'Sam', message: "Something went wrong. Please try again!" };
                this.message.push(msg2);
                this.updateChatText(chatBox);
            });
    }

    updateChatText(chatBox) {
        const chatMessages = chatBox.querySelector('.chatbox__messages');
        chatMessages.innerHTML = this.message
            .slice()
            .reverse()
            .map((item) =>
                item.name === "Sam"
                    ? `<div class="messages__item messages__item--visitor">${item.message}</div>`
                    : `<div class="messages__item messages__item--operator">${item.message}</div>`
            )
            .join('');
    }
}

const chatbox = new Chatbox();
chatbox.display();
