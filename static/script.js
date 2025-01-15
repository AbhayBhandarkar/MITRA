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
            showBlockedToast(); // Show a toast alert
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
        // Apply blocked/error/allowed
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

// Show a brief toast when a prompt is blocked
function showBlockedToast() {
    const toast = document.getElementById('toast-blocked');
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000); 
}

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

document.addEventListener('click', function(event) {
    if (event.target.closest('.chat-item')) {
        selectChatItem();
    }
});

// Initialize the chat on page load
document.addEventListener('DOMContentLoaded', () => {
    startNewChat();
});
