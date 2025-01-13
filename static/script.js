// script.js

// Initialize Confetti
const confettiSettings = { 
    particleCount: 100,
    spread: 70,
    origin: { y: 0.6 }
};
const triggerConfetti = () => {
    window.confetti(confettiSettings);
};

// Function to automatically resize the textarea based on input
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}

// Function to handle Enter and Shift+Enter key events
function handleKeyDown(event) {
    if (event.key === 'Enter') {
        if (!event.shiftKey) {
            event.preventDefault();
            submitMessage();
        }
    }
}

// Function to start a new chat (Clears the chat history)
function startNewChat() {
    // Clear messages container
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <h3>Welcome to MITRA 👋</h3>
            <p>I'm here to assist you with professional and responsible AI interactions.</p>
        </div>
    `;
    // Optionally, clear chat history or implement other logic
    // For example, you can clear localStorage or reset any chat-specific data
}

// Function to submit the user's message
async function submitMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message === "") {
        alert("Please enter a message.");
        return;
    }

    // Clear the input field
    userInput.value = "";
    autoResize(userInput);

    // Display the user's message
    addMessage(message, 'user');

    // Scroll to the latest message
    scrollToLatestMessage();

    try {
        // Disable the send button to prevent multiple submissions
        const sendButton = document.querySelector('.send-button');
        sendButton.disabled = true;

        // Display a loading spinner
        const messagesContainer = document.getElementById('messagesContainer');
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('message', 'bot', 'loading');
        loadingMessage.innerHTML = `
            <div class="avatar bot-avatar">M</div>
            <div class="message-content">
                <div class="loading-spinner"></div>
            </div>
        `;
        messagesContainer.appendChild(loadingMessage);
        scrollToLatestMessage();

        // Send the message to the backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: message })
        });

        const data = await response.json();

        // Remove the loading spinner
        messagesContainer.removeChild(loadingMessage);

        if (data.status === "blocked") {
            addMessage(data.error, 'bot', 'blocked');
        } else if (data.status === "allowed") {
            addMessage(data.response, 'bot');
            // Trigger confetti on successful response
            triggerConfetti();
        } else if (data.status === "error") {
            addMessage("An error occurred while processing your request.", 'bot', 'error');
            console.error("Error from backend:", data.error);
        }

    } catch (error) {
        console.error("Error:", error);
        addMessage("An unexpected error occurred.", 'bot', 'error');
    } finally {
        // Re-enable the send button
        const sendButton = document.querySelector('.send-button');
        sendButton.disabled = false;
        scrollToLatestMessage();
    }
}

// Function to add a message to the messages container
function addMessage(content, sender, status = 'allowed', isSystem = false) {
    const messagesContainer = document.getElementById('messagesContainer');

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    if (isSystem) {
        messageDiv.classList.add('system-message');
    } else {
        if (status === 'blocked' || status === 'error') {
            messageDiv.classList.add('blocked');
        } else {
            messageDiv.classList.add('allowed');
        }
    }

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('avatar');
    if (!isSystem) {
        avatarDiv.classList.add(sender === 'user' ? 'user-avatar' : 'bot-avatar');
        avatarDiv.textContent = sender === 'user' ? 'U' : 'M'; // 'U' for User, 'M' for MITRA
    }

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = content;

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    messagesContainer.appendChild(messageDiv);
}

// Function to scroll to the latest message
function scrollToLatestMessage() {
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Example Prompts Animation
const examplePrompts = [
    "How can I improve my team's productivity?",
    "Draft a professional email to a client.",
    "Provide a summary of the latest industry trends.",
    "Help me create a project timeline.",
    "What are the best practices for remote work?"
];

let currentPromptIndex = 0;
let currentCharIndex = 0;
let isDeleting = false;
const typingSpeed = 100; // milliseconds per character
const deletingSpeed = 50; // milliseconds per character
const pauseAfterTyping = 1500; // milliseconds after full prompt is typed
const pauseAfterDeleting = 500; // milliseconds after prompt is deleted

const promptTextElement = document.getElementById('promptText');
const cursorElement = document.getElementById('cursor');

function typeEffect() {
    const currentPrompt = examplePrompts[currentPromptIndex];
    if (!isDeleting) {
        // Typing phase
        promptTextElement.textContent = currentPrompt.substring(0, currentCharIndex + 1);
        currentCharIndex++;
        if (currentCharIndex === currentPrompt.length) {
            // Pause after typing
            isDeleting = true;
            setTimeout(typeEffect, pauseAfterTyping);
        } else {
            setTimeout(typeEffect, typingSpeed);
        }
    } else {
        // Deleting phase
        promptTextElement.textContent = currentPrompt.substring(0, currentCharIndex - 1);
        currentCharIndex--;
        if (currentCharIndex === 0) {
            isDeleting = false;
            currentPromptIndex = (currentPromptIndex + 1) % examplePrompts.length;
            setTimeout(typeEffect, pauseAfterDeleting);
        } else {
            setTimeout(typeEffect, deletingSpeed);
        }
    }
}

// Initialize typing effect on DOM load
document.addEventListener('DOMContentLoaded', () => {
    typeEffect();
});

// Sidebar Toggle Function
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('overlay');
    sidebar.classList.toggle('active');
    overlay.classList.toggle('active');
}

// Close sidebar when a chat item is clicked (optional)
function selectChatItem() {
    if (window.innerWidth <= 768) {
        toggleSidebar();
    }
}

// Attach selectChatItem to chat items
document.addEventListener('click', function(event) {
    if (event.target.closest('.chat-item')) {
        selectChatItem();
    }
});

// Initialize the chat on page load
document.addEventListener('DOMContentLoaded', () => {
    startNewChat();
});
