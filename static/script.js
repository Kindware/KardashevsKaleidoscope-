document.addEventListener('DOMContentLoaded', () => {
    const generateButton = document.getElementById('generate-plan');
    const resultsDiv = document.querySelector('.results');
    const advancedMetrics = document.querySelector('.advanced-metrics');
    
    generateButton.addEventListener('click', async () => {
        const strategy = document.getElementById('strategy').value;
        const includeColony = document.getElementById('include-colony').checked;
        const startYear = parseInt(document.getElementById('start-year').value);
        
        // Show loading state
        generateButton.textContent = 'Generating Plan...';
        generateButton.disabled = true;
        
        try {
            const response = await fetch('/generate_plan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy,
                    include_colony: includeColony,
                    start_year: startYear
                })
            });
            
            const data = await response.json();
            
            // Update metrics
            document.getElementById('total-time').textContent = `${data.metrics.total_time.toFixed(1)} years`;
            document.getElementById('completion-year').textContent = data.metrics.completion_year;
            document.getElementById('total-mass').textContent = formatMass(data.metrics.total_mass);
            document.getElementById('total-energy').textContent = `${data.metrics.total_energy.toFixed(1)} PW`;
            
            // Show/hide advanced metrics based on strategy
            if (['cosmic', 'dimensional', 'singularity', 'omniversal'].includes(strategy)) {
                advancedMetrics.style.display = 'block';
                document.getElementById('pocket-universes').textContent = data.metrics.pocket_universes.toLocaleString();
                document.getElementById('black-holes').textContent = data.metrics.black_holes.toLocaleString();
                document.getElementById('dimensions').textContent = data.metrics.dimensions.toLocaleString();
                document.getElementById('consciousness-uploads').textContent = data.metrics.consciousness_uploads.toLocaleString();
                document.getElementById('ai-evolution').textContent = data.metrics.ai_evolution.toFixed(1);
                document.getElementById('digital-realities').textContent = data.metrics.digital_realities.toLocaleString();
            } else {
                advancedMetrics.style.display = 'none';
            }
            
            // Update phases timeline
            const timelineDiv = document.getElementById('phases-timeline');
            timelineDiv.innerHTML = '';
            
            data.phases.forEach(phase => {
                const phaseDiv = document.createElement('div');
                phaseDiv.className = 'phase';
                phaseDiv.innerHTML = `
                    <h3>${phase.name}</h3>
                    <div class="details">
                        <div class="metric">
                            <span class="label">Time Required</span>
                            <span class="value">${phase.time.toFixed(1)} years</span>
                        </div>
                        <div class="metric">
                            <span class="label">Mass Required</span>
                            <span class="value">${formatMass(phase.mass)}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Energy Required</span>
                            <span class="value">${phase.energy.toFixed(1)} PW</span>
                        </div>
                    </div>
                    <div class="risks">
                        <h4>Risks:</h4>
                        <ul>
                            ${phase.risks.map(risk => `<li>${risk}</li>`).join('')}
                        </ul>
                    </div>
                `;
                timelineDiv.appendChild(phaseDiv);
            });
            
            // Update orbital grid
            const gridContainer = document.getElementById('orbital-grid');
            gridContainer.innerHTML = '';
            
            data.grid.forEach(row => {
                row.forEach(cell => {
                    const cellDiv = document.createElement('div');
                    cellDiv.className = 'grid-cell';
                    cellDiv.textContent = cell;
                    gridContainer.appendChild(cellDiv);
                });
            });
            
            // Show results
            resultsDiv.style.display = 'block';
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error generating plan. Please try again.');
        } finally {
            generateButton.textContent = 'Generate Plan';
            generateButton.disabled = false;
        }
    });
});

function formatMass(mass) {
    if (mass >= 1e30) {
        return `${(mass / 1e30).toFixed(2)} × 10³⁰ kg`;
    } else if (mass >= 1e27) {
        return `${(mass / 1e27).toFixed(2)} × 10²⁷ kg`;
    } else if (mass >= 1e24) {
        return `${(mass / 1e24).toFixed(2)} × 10²⁴ kg`;
    } else if (mass >= 1e21) {
        return `${(mass / 1e21).toFixed(2)} × 10²¹ kg`;
    } else {
        return `${mass.toExponential(2)} kg`;
    }
}

// Add particle effects to the background
function createParticles() {
    const container = document.querySelector('.container');
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animationDuration = `${Math.random() * 3 + 2}s`;
        particle.style.animationDelay = `${Math.random() * 2}s`;
        container.appendChild(particle);
    }
}

createParticles(); 