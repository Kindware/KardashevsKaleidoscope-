from flask import Flask, render_template, request, jsonify, Response
from dyson_sphere_planner import DysonSpherePlanner, DysonStrategy, DysonVisualization
import json
import time
from datetime import datetime
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    data = request.json
    strategy = DysonStrategy(data['strategy'].lower())
    include_colony = data.get('include_colony', False)
    start_year = int(data.get('start_year', 2024))
    
    # Collect all user variables into a dict
    user_vars = {
        'extraction_pct': float(data.get('extraction_pct', 1)),
        'transport_mass_mult': float(data.get('transport_mass_mult', 1)),
        'assembly_eff': float(data.get('assembly_eff', 1)),
        'thermal_eff': float(data.get('thermal_eff', 1)),
        'colony_init_pop': int(data.get('colony_init_pop', 1000)),
        'colony_growth_rate': float(data.get('colony_growth_rate', 2)),
        'colony_life_cap': int(data.get('colony_life_cap', 1000000)),
        'colony_resource_cap': int(data.get('colony_resource_cap', 500000)),
        'probe_speed': float(data.get('probe_speed', 0.1)),
        'sphere_construction_time': int(data.get('sphere_construction_time', 400)),
        'probe_build_time': int(data.get('probe_build_time', 50)),
        'quantum_network_size': int(data.get('quantum_network_size', 1000)),
        'exotic_matter_vol': float(data.get('exotic_matter_vol', 1000000)),
        'pocket_universes': int(data.get('pocket_universes', 0)),
        'dimensions_accessed': int(data.get('dimensions_accessed', 1)),
        'ai_evolution_level': float(data.get('ai_evolution_level', 2)),
    }
    
    planner = DysonSpherePlanner(strategy, include_colony, start_year, user_vars)
    
    # Generate the plan and collect all output
    plan_data = {
        'phases': [],
        'metrics': {},
        'grid': [],
        'warnings': [],
        'visualization': {
            'frames': [],
            'total_years': 0,
            'current_year': start_year
        }
    }
    
    # Capture the plan generation
    phases = []
    # Add advanced phases with event logs
    if strategy in [DysonStrategy.COSMIC, DysonStrategy.DIMENSIONAL, DysonStrategy.SINGULARITY, DysonStrategy.OMNIVERSAL]:
        req, events = planner.simulate_cosmic_engineering()
        phases.append(("Cosmic Engineering", req, events))
        req, events = planner.simulate_dimensional_engineering()
        phases.append(("Dimensional Engineering", req, events))
        req, events = planner.simulate_singularity_evolution()
        phases.append(("Singularity Evolution", req, events))
    # Add core phases (now with event logs)
    req = planner.simulate_material_extraction()
    phases.append(("Material Extraction", req, getattr(req, 'event_log', [])))
    req = planner.simulate_transport_system()
    phases.append(("Transport System", req, getattr(req, 'event_log', [])))
    req = planner.simulate_assembly()
    phases.append(("Assembly", req, getattr(req, 'event_log', [])))
    req = planner.simulate_thermal_management()
    phases.append(("Thermal Management", req, getattr(req, 'event_log', [])))
    if include_colony:
        req = planner.simulate_colony_setup()
        phases.append(("Colony Setup", req, getattr(req, 'event_log', [])))
    
    # Process phases
    total_time = 0
    current_year = start_year
    
    for phase_name, requirements, event_log in phases:
        phase_data = {
            'name': phase_name,
            'time': requirements.time_years,
            'mass': requirements.mass_tons,
            'energy': requirements.energy_petawatts,
            'risks': requirements.risks,
            'start_year': current_year,
            'end_year': current_year + requirements.time_years,
            'event_log': event_log if event_log else ["No major events occurred during this phase."]
        }
        plan_data['phases'].append(phase_data)
        total_time += requirements.time_years
        
        # Generate visualization frames for this phase
        for year in range(int(requirements.time_years)):
            grid = planner.generate_orbital_grid()
            # Ensure grid is a 2D array of single-character strings
            grid = [list(row) if isinstance(row, str) else row for row in grid]
            # Animate: reveal more tiles as progress increases
            progress = (year + 1) / requirements.time_years
            flat_indices = [(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] != ' ']
            reveal_count = int(progress * len(flat_indices))
            revealed = set(flat_indices[:reveal_count])
            animated_grid = [[' ' for _ in range(len(grid[0]))] for _ in range(len(grid))]
            for idx, (i, j) in enumerate(flat_indices):
                if (i, j) in revealed:
                    animated_grid[i][j] = grid[i][j]
            frame = {
                'year': current_year + year,
                'phase': phase_name,
                'progress': progress,
                'grid': animated_grid
            }
            plan_data['visualization']['frames'].append(frame)
            
        current_year += requirements.time_years
    
    # Calculate total mass and energy
    total_mass = sum(phase['mass'] for phase in plan_data['phases'])
    total_energy = sum(phase['energy'] for phase in plan_data['phases'])
    
    # Add metrics
    plan_data['metrics'] = {
        'total_time': total_time,
        'completion_year': start_year + total_time,
        'strategy': strategy.value,
        'include_colony': include_colony,
        'total_mass': total_mass,
        'total_energy': total_energy
    }
    
    # Add advanced metrics if applicable
    if strategy in [DysonStrategy.COSMIC, DysonStrategy.DIMENSIONAL, 
                   DysonStrategy.SINGULARITY, DysonStrategy.OMNIVERSAL]:
        # Creative, plausible advanced metrics
        pocket_universes = user_vars.get('pocket_universes', 0) + random.randint(1, 5)
        black_holes = random.randint(1, 7) + (1 if strategy == DysonStrategy.COSMIC else 0)
        dimensions = user_vars.get('dimensions_accessed', 1)
        consciousness_uploads = 1000 + random.randint(0, 500)
        ai_evolution = user_vars.get('ai_evolution_level', 2) + random.uniform(0, 2)
        digital_realities = 1 + random.randint(0, 4)
        plan_data['metrics'].update({
            'pocket_universes': pocket_universes,
            'black_holes': black_holes,
            'dimensions': dimensions,
            'consciousness_uploads': consciousness_uploads,
            'ai_evolution': round(ai_evolution, 2),
            'digital_realities': digital_realities
        })
    
    # Set visualization metadata
    plan_data['visualization']['total_years'] = total_time
    plan_data['visualization']['current_year'] = start_year
    
    return jsonify(plan_data)

@app.route('/visualization/stream')
def visualization_stream():
    """Stream visualization updates"""
    def generate():
        while True:
            # Get current visualization state
            data = {
                'year': current_year,
                'grid': current_grid,
                'progress': current_progress
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)  # Update every 100ms
            
    return Response(generate(), mimetype='text/event-stream')

@app.route('/check_alien_contacts', methods=['POST'])
def check_alien_contacts():
    data = request.json
    strategy = DysonStrategy(data['strategy'].lower())
    
    planner = DysonSpherePlanner(strategy, include_colony=True, start_year=2024)
    planner._simulate_alien_encounters()
    
    return jsonify({
        'contacts': [
            {
                'species': contact.species.value,
                'motivation': contact.motivation.value,
                'tech_level': contact.tech_level,
                'first_contact_year': contact.first_contact_year
            }
            for contact in planner.alien_contacts
        ]
    })

@app.route('/simulate_evolution', methods=['POST'])
def simulate_evolution():
    data = request.json
    strategy = DysonStrategy(data['strategy'].lower())
    years = data['years']
    
    planner = DysonSpherePlanner(strategy, include_colony=True, start_year=2024)
    planner.universe_evolution.simulate_time_step(years)
    
    return jsonify({
        'events': [
            {
                'year': event.year,
                'event_type': event.event_type,
                'description': event.description,
                'impact': event.impact
            }
            for event in planner.universe_evolution.universe_events
            if event.impact > 0.5
        ]
    })

@app.route('/generate_custom_structure', methods=['POST'])
def generate_custom_structure():
    data = request.json
    
    planner = DysonSpherePlanner(DysonStrategy.SWARM, include_colony=True, start_year=2024)
    structure = planner.custom_editor.create_custom_structure(
        name=f"Custom-{datetime.now().isoformat()}",
        collector_pattern=data['collector_pattern'],
        habitat_pattern=data['habitat_pattern'],
        transport_pattern=data['transport_pattern'],
        thermal_pattern=data['thermal_pattern']
    )
    
    return jsonify({
        'structure': {
            'name': structure.name,
            'efficiency_score': structure.efficiency_score,
            'aesthetic_score': structure.aesthetic_score,
            'resource_requirements': {
                'mass_tons': structure.resource_requirements.mass_tons,
                'energy_petawatts': structure.resource_requirements.energy_petawatts,
                'time_years': structure.resource_requirements.time_years,
                'risks': structure.resource_requirements.risks
            }
        }
    })

@app.route('/initialize_virtual_civilization', methods=['POST'])
def initialize_virtual_civilization():
    data = request.json
    strategy = DysonStrategy(data['strategy'].lower())
    
    planner = DysonSpherePlanner(strategy, include_colony=True, start_year=2024)
    planner._initialize_virtual_civilization()
    
    return jsonify({
        'events': [
            {
                'year': event.year,
                'event_type': event.event_type,
                'description': event.description,
                'impact': event.impact
            }
            for event in planner.digital_events
            if event.impact > 0.3
        ]
    })

@app.route('/upload_consciousness', methods=['POST'])
def upload_consciousness():
    data = request.json
    count = data['count']
    
    planner = DysonSpherePlanner(DysonStrategy.SINGULARITY, include_colony=True, start_year=2024)
    events = []
    
    for _ in range(count):
        consciousness = planner.virtual_civilization.upload_consciousness("human")
        events.extend(planner.virtual_civilization.digital_events[-1:])
    
    return jsonify({
        'events': [
            {
                'year': event.year,
                'event_type': event.event_type,
                'description': event.description,
                'impact': event.impact
            }
            for event in events
        ]
    })

@app.route('/create_reality', methods=['POST'])
def create_reality():
    data = request.json
    
    planner = DysonSpherePlanner(DysonStrategy.SINGULARITY, include_colony=True, start_year=2024)
    reality = planner.virtual_civilization.create_virtual_reality(
        name=data['name'],
        size=data['size'],
        time_dilation=data['time_dilation']
    )
    
    return jsonify({
        'events': [
            {
                'year': event.year,
                'event_type': event.event_type,
                'description': event.description,
                'impact': event.impact
            }
            for event in planner.virtual_civilization.digital_events[-1:]
        ]
    })

@app.route('/generate_story', methods=['POST'])
def generate_story():
    data = request.json
    strategy = DysonStrategy[data['strategy'].upper()]
    start_year = data['start_year']
    include_colony = data['include_colony']
    
    # Create a new simulation instance
    simulation = DysonSpherePlanner(strategy, include_colony, start_year)
    
    # Run the simulation to get the story
    results = simulation.run_simulation()
    
    return jsonify({'story': results['story']})

if __name__ == '__main__':
    app.run(debug=True, port=5005) 