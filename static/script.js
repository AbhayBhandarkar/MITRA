/* script.js - Enhanced client-side interactions for MITRA with conversation context */

const confettiSettings = {
    particleCount: 100,
    spread: 70,
    origin: { y: 0.6 }
};

const triggerConfetti = () => {
    window.confetti(confettiSettings);
};

let conversationHistory = "";  // Maintain conversation history as a string

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        submitMessage();
    }
}

function startNewChat() {
    conversationHistory = "";  // Reset history for a new chat
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <h3>Welcome to MITRA 👋</h3>
            <p>I'm here to assist you with professional and responsible AI interactions.</p>
        </div>`;
}

async function submitMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message === "") {
        alert("Please enter a message.");
        return;
    }

    // Append user's message to the conversation history.
    conversationHistory += " " + message;

    userInput.value = "";
    autoResize(userInput);
    addMessage(message, 'user');
    scrollToLatestMessage();

    try {
        const sendButton = document.querySelector('.send-button');
        sendButton.disabled = true;

        const messagesContainer = document.getElementById('messagesContainer');
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('message', 'bot', 'loading');
        loadingMessage.innerHTML = `
            <div class="avatar bot-avatar">M</div>
            <div class="message-content"><div class="loading-spinner"></div></div>`;
        messagesContainer.appendChild(loadingMessage);
        scrollToLatestMessage();

        // Send both the prompt and the conversation history to the backend.
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: message, history: conversationHistory })
        });

        const data = await response.json();
        messagesContainer.removeChild(loadingMessage);

        if (data.status === "blocked") {
            addMessage(data.error, 'bot', 'blocked');
            showBlockedToast();
        } else if (data.status === "allowed") {
            addMessage(data.response, 'bot');
            triggerConfetti();
        } else if (data.status === "error") {
            addMessage("An error occurred while processing your request.", 'bot', 'error');
            console.error("Error from backend:", data.error);
        }
        // Append bot response to conversation history.
        if(data.response) {
            conversationHistory += " " + data.response;
        }
    } catch (error) {
        console.error("Error:", error);
        addMessage("An unexpected error occurred.", 'bot', 'error');
    } finally {
        const sendButton = document.querySelector('.send-button');
        sendButton.disabled = false;
        scrollToLatestMessage();
    }
}

function addMessage(content, sender, status = 'allowed', isSystem = false) {
    const messagesContainer = document.getElementById('messagesContainer');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    if (isSystem) {
        messageDiv.classList.add('system-message');
    } else {
        messageDiv.classList.add(status === 'blocked' || status === 'error' ? 'blocked' : 'allowed');
    }

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('avatar');
    if (!isSystem) {
        avatarDiv.classList.add(sender === 'user' ? 'user-avatar' : 'bot-avatar');
        avatarDiv.textContent = sender === 'user' ? 'U' : 'M';
    }

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = content;

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
}

function scrollToLatestMessage() {
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showBlockedToast() {
    const toast = document.getElementById('toast-blocked');
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('overlay');
    sidebar.classList.toggle('active');
    overlay.classList.toggle('active');
}

function selectChatItem() {
    if (window.innerWidth <= 768) {
        toggleSidebar();
    }
}

document.addEventListener('click', function(event) {
    if (event.target.closest('.chat-item')) {
        selectChatItem();
    }
});

document.addEventListener('DOMContentLoaded', () => {
    startNewChat();
});
