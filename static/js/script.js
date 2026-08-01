document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const spinner = predictBtn.querySelector('.spinner');
    const resultSection = document.getElementById('result-section');
    
    // Result elements
    const resultCard = document.getElementById('result-section');
    const resultIcon = document.getElementById('result-icon');
    const resultLabel = document.getElementById('result-label');
    const probabilityValue = document.getElementById('probability-value');
    const probabilityProgress = document.getElementById('probability-progress');
    const recommendationText = document.getElementById('recommendation-text');
    
    // Diagnostic elements
    const detailDistance = document.getElementById('detail-distance');
    const detailHour = document.getElementById('detail-hour');
    const detailDay = document.getElementById('detail-day');
    const detailMerchant = document.getElementById('detail-merchant');
    
    // Form submission handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // 1. Validate inputs before submitting
        const amt = parseFloat(document.getElementById('amt').value);
        const age = parseInt(document.getElementById('age').value);
        const cityPop = parseInt(document.getElementById('city_pop').value);
        
        if (isNaN(amt) || amt <= 0) {
            alert('Please enter a valid transaction amount greater than 0.');
            return;
        }
        if (isNaN(age) || age < 18 || age > 120) {
            alert('Please enter a valid age between 18 and 120.');
            return;
        }
        if (isNaN(cityPop) || cityPop < 0) {
            alert('Please enter a valid city population.');
            return;
        }
        
        // 2. Set loading state
        setLoading(true);
        
        // Prepare request payload
        const payload = {
            amt: amt,
            category: document.getElementById('category').value,
            merchant: document.getElementById('merchant').value,
            age: age,
            gender: document.getElementById('gender').value,
            state: document.getElementById('state').value,
            city_pop: cityPop
        };
        
        try {
            // 3. Post request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                // 4. Render prediction results with animations
                displayResults(result);
            } else {
                alert(`Error: ${result.message || 'Verification failed.'}`);
            }
        } catch (error) {
            console.error('Submission error:', error);
            alert('A network error occurred. Please ensure the backend server is running.');
        } finally {
            setLoading(false);
        }
    });
    
    // Manage Loading State
    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            resultSection.classList.add('hidden');
        } else {
            predictBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }
    
    // Display results with dynamic styling and progress bar animations
    function displayResults(data) {
        // Reset classes
        resultCard.classList.remove('legitimate', 'fraud');
        
        // Determine type of result card
        const isFraud = data.prediction === 'fraud';
        resultCard.classList.add(isFraud ? 'fraud' : 'legitimate');
        
        // Set classification labels
        resultLabel.textContent = data.label;
        recommendationText.textContent = data.recommendation;
        
        // Set icon dynamically
        if (isFraud) {
            resultIcon.setAttribute('data-lucide', 'shield-alert');
        } else {
            resultIcon.setAttribute('data-lucide', 'shield-check');
        }
        // Force Lucide to update the specific icon
        lucide.createIcons({
            attrs: {
                class: 'result-badge-icon'
            },
            nameAttr: 'data-lucide'
        });
        
        // Set Diagnostics
        detailDistance.textContent = `${data.details.distance_km} km`;
        detailHour.textContent = `${data.details.hour}:00`;
        
        // Map day number to name
        const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        detailDay.textContent = days[data.details.day_of_week] || '-';
        detailMerchant.textContent = data.details.merchant_used;
        
        // Show result card
        resultSection.classList.remove('hidden');
        
        // Animate progress bar & value counter
        const probabilityPct = data.probability * 100;
        
        // Fill bar with delay for transition effect
        setTimeout(() => {
            probabilityProgress.style.width = `${probabilityPct}%`;
        }, 100);
        
        // Animate counter
        animateCounter(probabilityValue, probabilityPct);
    }
    
    // Smooth number counter animation
    function animateCounter(element, targetValue) {
        let currentValue = 0;
        const duration = 1000; // 1 second
        const steps = 60;
        const stepValue = targetValue / steps;
        const stepTime = duration / steps;
        
        const interval = setInterval(() => {
            currentValue += stepValue;
            if (currentValue >= targetValue) {
                element.textContent = `${targetValue.toFixed(2)}%`;
                clearInterval(interval);
            } else {
                element.textContent = `${currentValue.toFixed(2)}%`;
            }
        }, stepTime);
    }
});

// Diagnostic Section Toggle Action
function toggleDetails() {
    const header = document.querySelector('.accordion-header');
    const content = document.getElementById('accordion-content');
    
    header.classList.toggle('active');
    content.classList.toggle('hidden');
}
