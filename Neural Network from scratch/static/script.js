const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultDigit = document.getElementById('resultDigit');
const resultConfidence = document.getElementById('resultConfidence');
const confidenceFill = document.getElementById('confidenceFill');

// Drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Setup Canvas
function setupCanvas() {
    // Fill with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20; // Thick enough for 28x28 downscaling
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

setupCanvas();

// Event Listeners for Mouse
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
});

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

// Event Listeners for Touch
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    lastX = touch.clientX - rect.left;
    lastY = touch.clientY - rect.top;
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;

    drawParams(x, y);
});

canvas.addEventListener('touchend', () => isDrawing = false);

function draw(e) {
    if (!isDrawing) return;
    drawParams(e.offsetX, e.offsetY);
}

function drawParams(x, y) {
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

// Clear Canvas
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDigit.innerText = '-';
    resultConfidence.innerText = '0.00%';
    confidenceFill.style.width = '0%';
});

// Predict
predictBtn.addEventListener('click', async () => {
    // Get image data
    const dataURL = canvas.toDataURL('image/png');

    // Show loading state
    resultDigit.innerText = '...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: dataURL })
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Update UI
        resultDigit.innerText = data.digit;
        resultConfidence.innerText = data.confidence + '%';
        confidenceFill.style.width = data.confidence + '%';

    } catch (error) {
        console.error('Error:', error);
        resultDigit.innerText = '?';
    }
});
