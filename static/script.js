document.addEventListener('DOMContentLoaded', () => {
    const generateButton = document.getElementById('generate-plan');
    const resultsDiv = document.querySelector('.results');
    const advancedMetrics = document.querySelector('.advanced-metrics');
    const resetButton = document.getElementById('reset-variables');
    const extractionPct = document.getElementById('extraction_pct');
    const extractionPctVal = document.getElementById('extraction_pct_val');
    
    // Show extraction percentage live
    extractionPct.addEventListener('input', () => {
        extractionPctVal.textContent = extractionPct.value;
    });

    // Reset to default values
    resetButton.addEventListener('click', () => {
        document.getElementById('extraction_pct').value = 1;
        extractionPctVal.textContent = 1;
        document.getElementById('transport_mass_mult').value = 1;
        document.getElementById('assembly_eff').value = 1;
        document.getElementById('thermal_eff').value = 1;
        document.getElementById('colony_init_pop').value = 1000;
        document.getElementById('colony_growth_rate').value = 2;
        document.getElementById('colony_life_cap').value = 1000000;
        document.getElementById('colony_resource_cap').value = 500000;
        document.getElementById('probe_speed').value = 0.1;
        document.getElementById('sphere_construction_time').value = 400;
        document.getElementById('probe_build_time').value = 50;
        document.getElementById('quantum_network_size').value = 1000;
        document.getElementById('exotic_matter_vol').value = 1000000;
        document.getElementById('pocket_universes').value = 0;
        document.getElementById('dimensions_accessed').value = 1;
        document.getElementById('ai_evolution_level').value = 2;
    });

    generateButton.addEventListener('click', async () => {
        const strategy = document.getElementById('strategy').value;
        const includeColony = document.getElementById('include-colony').checked;
        const startYear = parseInt(document.getElementById('start-year').value);
        
        // Collect all variable values
        const variables = {
            extraction_pct: parseFloat(document.getElementById('extraction_pct').value),
            transport_mass_mult: parseFloat(document.getElementById('transport_mass_mult').value),
            assembly_eff: parseFloat(document.getElementById('assembly_eff').value),
            thermal_eff: parseFloat(document.getElementById('thermal_eff').value),
            colony_init_pop: parseInt(document.getElementById('colony_init_pop').value),
            colony_growth_rate: parseFloat(document.getElementById('colony_growth_rate').value),
            colony_life_cap: parseInt(document.getElementById('colony_life_cap').value),
            colony_resource_cap: parseInt(document.getElementById('colony_resource_cap').value),
            probe_speed: parseFloat(document.getElementById('probe_speed').value),
            sphere_construction_time: parseInt(document.getElementById('sphere_construction_time').value),
            probe_build_time: parseInt(document.getElementById('probe_build_time').value),
            quantum_network_size: parseInt(document.getElementById('quantum_network_size').value),
            exotic_matter_vol: parseFloat(document.getElementById('exotic_matter_vol').value),
            pocket_universes: parseInt(document.getElementById('pocket_universes').value),
            dimensions_accessed: parseInt(document.getElementById('dimensions_accessed').value),
            ai_evolution_level: parseFloat(document.getElementById('ai_evolution_level').value)
        };
        
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
                    start_year: startYear,
                    ...variables
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
                    <div class="event-log">
                        <h4>Event Log:</h4>
                        ${phase.event_log.map(ev => `<div class='event-log-entry'>${ev}</div>`).join('')}
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
            
            // Initialize visualization
            setNewPlanData(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error generating plan. Please try again.');
        } finally {
            generateButton.textContent = 'Generate Plan';
            generateButton.disabled = false;
        }
    });

    // Alien Diplomacy
    const checkAlienContactsBtn = document.getElementById('check-alien-contacts');
    const alienContactsList = document.getElementById('alien-contacts-list');
    
    checkAlienContactsBtn.addEventListener('click', async () => {
        const response = await fetch('/check_alien_contacts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy: document.getElementById('strategy').value })
        });
        
        const data = await response.json();
        displayAlienContacts(data.contacts);
    });
    
    function displayAlienContacts(contacts) {
        alienContactsList.innerHTML = contacts.map(contact => `
            <div class="event">
                <div class="year">Year ${contact.first_contact_year}</div>
                <div class="description">
                    First contact with ${contact.species} civilization.
                    Tech Level: ${(contact.tech_level * 100).toFixed(1)}%
                    Motivation: ${contact.motivation}
                </div>
            </div>
        `).join('');
    }
    
    // Universe Evolution
    const simulateEvolutionBtn = document.getElementById('simulate-evolution');
    const evolutionYearsInput = document.getElementById('evolution-years');
    const evolutionYearsValue = document.getElementById('evolution-years-value');
    const universeEventsList = document.getElementById('universe-events-list');
    
    evolutionYearsInput.addEventListener('input', (e) => {
        evolutionYearsValue.textContent = `${Number(e.target.value).toLocaleString()} years`;
    });
    
    simulateEvolutionBtn.addEventListener('click', async () => {
        const response = await fetch('/simulate_evolution', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                years: parseInt(evolutionYearsInput.value),
                strategy: document.getElementById('strategy').value
            })
        });
        
        const data = await response.json();
        displayUniverseEvents(data.events);
    });
    
    function displayUniverseEvents(events) {
        universeEventsList.innerHTML = events.map(event => `
            <div class="event">
                <div class="year">Year ${event.year}</div>
                <div class="description">
                    ${event.event_type}: ${event.description}
                    Impact: ${(event.impact * 100).toFixed(1)}%
                </div>
            </div>
        `).join('');
    }
    
    // Custom Megastructure Editor
    const generateCustomBtn = document.getElementById('generate-custom');
    const customStructuresList = document.getElementById('custom-structures-list');
    
    generateCustomBtn.addEventListener('click', async () => {
        const response = await fetch('/generate_custom_structure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                collector_pattern: document.getElementById('collector-pattern').value,
                habitat_pattern: document.getElementById('habitat-pattern').value,
                transport_pattern: document.getElementById('transport-pattern').value,
                thermal_pattern: document.getElementById('thermal-pattern').value
            })
        });
        
        const data = await response.json();
        displayCustomStructure(data.structure);
    });
    
    function displayCustomStructure(structure) {
        const structureCard = document.createElement('div');
        structureCard.className = 'structure-card';
        structureCard.innerHTML = `
            <h3>${structure.name}</h3>
            <div class="metrics">
                <div class="metric">
                    <span class="label">Efficiency:</span>
                    <span class="value">${(structure.efficiency_score * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="label">Aesthetic:</span>
                    <span class="value">${(structure.aesthetic_score * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="label">Mass Required:</span>
                    <span class="value">${structure.resource_requirements.mass_tons.toExponential(2)} tons</span>
                </div>
                <div class="metric">
                    <span class="label">Energy Required:</span>
                    <span class="value">${structure.resource_requirements.energy_petawatts.toExponential(2)} PW</span>
                </div>
            </div>
        `;
        customStructuresList.prepend(structureCard);
    }
    
    // Virtual Civilization
    const initializeVirtualBtn = document.getElementById('initialize-virtual');
    const uploadConsciousnessBtn = document.getElementById('upload-consciousness');
    const createRealityBtn = document.getElementById('create-reality');
    const consciousnessCountInput = document.getElementById('consciousness-count');
    const virtualEventsList = document.getElementById('virtual-events-list');
    
    initializeVirtualBtn.addEventListener('click', async () => {
        const response = await fetch('/initialize_virtual_civilization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                strategy: document.getElementById('strategy').value
            })
        });
        
        const data = await response.json();
        displayVirtualEvents(data.events);
    });
    
    uploadConsciousnessBtn.addEventListener('click', async () => {
        const response = await fetch('/upload_consciousness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                count: parseInt(consciousnessCountInput.value)
            })
        });
        
        const data = await response.json();
        displayVirtualEvents(data.events);
    });
    
    createRealityBtn.addEventListener('click', async () => {
        const response = await fetch('/create_reality', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: `Reality-${Date.now()}`,
                size: 1000.0,
                time_dilation: 10.0
            })
        });
        
        const data = await response.json();
        displayVirtualEvents(data.events);
    });
    
    function displayVirtualEvents(events) {
        virtualEventsList.innerHTML = events.map(event => `
            <div class="event">
                <div class="year">Year ${event.year}</div>
                <div class="description">
                    ${event.event_type}: ${event.description}
                    Impact: ${(event.impact * 100).toFixed(1)}%
                </div>
            </div>
        `).join('');
    }
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

let isPlaying = false;
let currentFrame = 0;
let animationSpeed = 1;

function updateVisualization(data, frameIndexOverride) {
    if (!data || !data.visualization || !data.visualization.frames || data.visualization.frames.length === 0) {
        const gridContainer = document.getElementById('orbital-grid');
        if (gridContainer) gridContainer.innerHTML = '<div style="color:red;">No visualization data available.</div>';
        const yearDisplay = document.getElementById('current-year');
        if (yearDisplay) yearDisplay.textContent = '-';
        const progressBar = document.getElementById('construction-progress');
        if (progressBar) progressBar.style.width = '0%';
        return;
    }
    const visualization = data.visualization;
    const gridContainer = document.getElementById('orbital-grid');
    const yearDisplay = document.getElementById('current-year');
    const progressBar = document.getElementById('construction-progress');

    // Calculate frame index safely
    const startYear = (data.metrics && data.metrics.start_year) ? data.metrics.start_year : 2024;
    let frameIndex = typeof frameIndexOverride === 'number' ? frameIndexOverride : currentFrame;
    frameIndex = Math.max(0, Math.min(frameIndex, visualization.frames.length - 1));

    // Update year display
    if (yearDisplay) yearDisplay.textContent = visualization.frames[frameIndex].year;

    // Update progress bar
    if (progressBar) {
        const progress = (frameIndex) / (visualization.frames.length - 1) * 100;
        progressBar.style.width = `${progress}%`;
    }

    // Update grid
    gridContainer.innerHTML = '';
    const grid = visualization.frames[frameIndex].grid;
    if (!grid) {
        gridContainer.innerHTML = '<div style="color:red;">No grid data for this year.</div>';
        return;
    }

    // Patch: Render as a flat CSS grid (no .grid-row)
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[i].length; j++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.textContent = grid[i][j];
            switch (grid[i][j]) {
                case 'O': cell.classList.add('collector'); break;
                case '#': cell.classList.add('habitat'); break;
                case 'R': case 'r': cell.classList.add('resource'); break;
                case 'T': cell.classList.add('high-traffic'); break;
                case 't': cell.classList.add('medium-traffic'); break;
                case 'H': case 'C': cell.classList.add('thermal'); break;
            }
            gridContainer.appendChild(cell);
        }
    }
}

// Add visualization controls
document.addEventListener('DOMContentLoaded', function() {
    const controls = document.createElement('div');
    controls.className = 'visualization-controls';
    controls.innerHTML = `
        <button id="play-pause" class="control-button" disabled>Play</button>
        <button id="step-backward" class="control-button" disabled>◀</button>
        <button id="step-forward" class="control-button" disabled>▶</button>
        <button id="take-screenshot" class="control-button" disabled>Screenshot</button>
        <div class="speed-control">
            <label for="speed">Speed:</label>
            <input type="range" id="speed" min="1" max="10" value="1">
        </div>
    `;
    document.querySelector('.orbital-grid').insertBefore(controls, document.getElementById('orbital-grid'));

    const playPauseBtn = document.getElementById('play-pause');
    const stepBackBtn = document.getElementById('step-backward');
    const stepFwdBtn = document.getElementById('step-forward');
    const screenshotBtn = document.getElementById('take-screenshot');
    const speedInput = document.getElementById('speed');

    playPauseBtn.addEventListener('click', function() {
        if (!window.currentPlanData || !window.currentPlanData.visualization) return;
        isPlaying = !isPlaying;
        this.textContent = isPlaying ? 'Pause' : 'Play';
        if (isPlaying) {
            animate();
        }
    });
    stepBackBtn.addEventListener('click', function() {
        if (!window.currentPlanData || !window.currentPlanData.visualization) return;
        if (currentFrame > 0) {
            currentFrame--;
            updateVisualization(window.currentPlanData, currentFrame);
        }
    });
    stepFwdBtn.addEventListener('click', function() {
        if (!window.currentPlanData || !window.currentPlanData.visualization) return;
        if (currentFrame < window.currentPlanData.visualization.frames.length - 1) {
            currentFrame++;
            updateVisualization(window.currentPlanData, currentFrame);
        }
    });
    screenshotBtn.addEventListener('click', function() {
        if (!window.currentPlanData || !window.currentPlanData.visualization) return;
        const grid = document.getElementById('orbital-grid');
        html2canvas(grid).then(canvas => {
            const link = document.createElement('a');
            link.download = `dyson-sphere-year-${window.currentPlanData.visualization.frames[currentFrame].year}.png`;
            link.href = canvas.toDataURL();
            link.click();
        });
    });
    speedInput.addEventListener('input', function() {
        animationSpeed = parseInt(this.value);
    });

    function animate() {
        if (!isPlaying || !window.currentPlanData || !window.currentPlanData.visualization) return;
        if (currentFrame < window.currentPlanData.visualization.frames.length - 1) {
            currentFrame++;
            updateVisualization(window.currentPlanData, currentFrame);
            setTimeout(animate, 1000 / animationSpeed);
        } else {
            isPlaying = false;
            playPauseBtn.textContent = 'Play';
        }
    }

    // Patch: Enable controls only when a plan is loaded
    window.enableVisualizationControls = function() {
        playPauseBtn.disabled = false;
        stepBackBtn.disabled = false;
        stepFwdBtn.disabled = false;
        screenshotBtn.disabled = false;
    };
});

// Patch: When a new plan is generated, reset currentFrame and update visualization, and enable controls
window.setNewPlanData = function(data) {
    window.currentPlanData = data;
    currentFrame = 0;
    updateVisualization(data, 0);
    if (window.enableVisualizationControls) window.enableVisualizationControls();
};

// Sci-Fi Tooltip accessibility: show tooltip on focus/blur for keyboard users
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.tooltip-icon').forEach(function(icon) {
        icon.addEventListener('focus', function() {
            icon.classList.add('focus');
        });
        icon.addEventListener('blur', function() {
            icon.classList.remove('focus');
        });
        // Make icon focusable
        icon.setAttribute('tabindex', '0');
    });
});

// Sci-Fi Feature Tabs switching
function enableFeatureTabs() {
    const tabs = document.querySelectorAll('.feature-tabs .tab');
    const sections = {
        alien: document.getElementById('tab-alien'),
        evolution: document.getElementById('tab-evolution'),
        custom: document.getElementById('tab-custom'),
        virtual: document.getElementById('tab-virtual')
    };
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            Object.values(sections).forEach(sec => sec.style.display = 'none');
            sections[tab.dataset.tab].style.display = '';
        });
        tab.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                tab.click();
            }
        });
    });
}
document.addEventListener('DOMContentLoaded', enableFeatureTabs); 