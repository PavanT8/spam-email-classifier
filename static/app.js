document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const emailContent = document.getElementById('email-content');
    const resultContainer = document.getElementById('result-container');
    const resultCard = document.querySelector('.result-card');
    const resultLabel = document.getElementById('result-label');
    const resultConfidence = document.getElementById('result-confidence');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    analyzeBtn.addEventListener('click', async () => {
        const text = emailContent.value.trim();
        
        if (!text) {
            showError("Please enter some email content to analyze.");
            return;
        }

        // Reset UI
        hideResult();
        hideError();
        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Expected response format: { label: "spam" | "ham", probability: 0.95 }
            // or { prediction: "spam", confidence: 0.95 }
            
            const label = (data.label || data.prediction || "unknown").toLowerCase();
            const probability = data.probability !== undefined ? data.probability : 
                              (data.confidence !== undefined ? data.confidence : null);
            
            if (label === "unknown" || probability === null) {
                 throw new Error("Invalid response format from API");
            }

            showResult(label, probability);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            showError("Failed to analyze email. Make sure the server is running.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="loader"></span> Analyzing...';
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Email';
        }
    }

    function showResult(label, probability) {
        resultContainer.classList.remove('hidden');
        
        const isSpam = label === 'spam';
        
        resultCard.className = 'result-card'; // Reset classes
        resultCard.classList.add(isSpam ? 'is-spam' : 'is-ham');
        
        resultLabel.textContent = isSpam ? 'Spam' : 'Not Spam (Ham)';
        
        // Format probability to percentage
        const percentage = (probability * 100).toFixed(1);
        resultConfidence.textContent = `Confidence: ${percentage}%`;
    }

    function hideResult() {
        resultContainer.classList.add('hidden');
    }

    function showError(msg) {
        errorContainer.classList.remove('hidden');
        errorMessage.textContent = msg;
    }

    function hideError() {
        errorContainer.classList.add('hidden');
    }
});
