from flask import Flask, render_template, request, jsonify, Response
from dyson_sphere_planner import DysonSpherePlanner, DysonStrategy, DysonVisualization
import json
import time
from datetime import datetime
import random
import math

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
        'probe_build_time': int(data.get('probe_build_time', 50)),
        'quantum_network_size': int(data.get('quantum_network_size', 1000)),
        'exotic_matter_vol': float(data.get('exotic_matter_vol', 1000000)),
        'pocket_universes': int(data.get('pocket_universes', 0)),
        'dimensions_accessed': int(data.get('dimensions_accessed', 1)),
        'ai_evolution_level': float(data.get('ai_evolution_level', 2)),
        'construction_speed_mod': float(data.get('construction_speed_mod', 1)),
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
    
    # --- Overlapping/Parallel Phases Refactor ---
    # 1. Calculate phase durations and overlap rules
    # Overlap rules: (percentages can be tweaked for realism)
    # - Transport starts at 30% of Extraction
    # - Assembly starts at 30% of Transport
    # - Thermal starts at 30% of Assembly
    # - Colony starts at 50% of Assembly (if enabled)

    # Simulate all phases to get durations
    extraction_req = planner.simulate_material_extraction()
    extraction_time = int(extraction_req.time_years)
    transport_req = planner.simulate_transport_system()
    transport_time = int(transport_req.time_years)
    assembly_req = planner.simulate_assembly()
    assembly_time = int(assembly_req.time_years)
    thermal_req = planner.simulate_thermal_management()
    thermal_time = int(thermal_req.time_years)
    if include_colony:
        colony_req = planner.simulate_colony_setup()
        colony_time = int(colony_req.time_years)
    else:
        colony_req = None
        colony_time = 0

    # Advanced phases (if any)
    advanced_phases = []
    if strategy in [DysonStrategy.COSMIC, DysonStrategy.DIMENSIONAL, DysonStrategy.SINGULARITY, DysonStrategy.OMNIVERSAL]:
        cosmic_req, cosmic_events = planner.simulate_cosmic_engineering()
        dimensional_req, dimensional_events = planner.simulate_dimensional_engineering()
        singularity_req, singularity_events = planner.simulate_singularity_evolution()
        advanced_phases = [
            ("Cosmic Engineering", cosmic_req, cosmic_events),
            ("Dimensional Engineering", dimensional_req, dimensional_events),
            ("Singularity Evolution", singularity_req, singularity_events)
        ]

    # Calculate start/end years for each phase (with overlap)
    extraction_start = start_year
    extraction_end = extraction_start + extraction_time
    transport_start = extraction_start + int(0.3 * extraction_time)
    transport_end = transport_start + transport_time
    assembly_start = transport_start + int(0.3 * transport_time)
    assembly_end = assembly_start + assembly_time
    thermal_start = assembly_start + int(0.3 * assembly_time)
    thermal_end = thermal_start + thermal_time
    if include_colony:
        colony_start = assembly_start + int(0.5 * assembly_time)
        colony_end = colony_start + colony_time
    else:
        colony_start = colony_end = 0

    # Advanced phase offsets (after core phases)
    adv_offset = max(extraction_end, transport_end, assembly_end, thermal_end, colony_end)
    adv_phase_schedules = []
    for i, (name, req, events) in enumerate(advanced_phases):
        adv_start = adv_offset + i * 100  # Stagger advanced phases
        adv_end = adv_start + int(req.time_years)
        adv_phase_schedules.append((name, req, events, adv_start, adv_end))

    # Collect all phases with their schedules
    phase_schedules = [
        ("Material Extraction", extraction_req, getattr(extraction_req, 'event_log', []), extraction_start, extraction_end),
        ("Transport System", transport_req, getattr(transport_req, 'event_log', []), transport_start, transport_end),
        ("Assembly", assembly_req, getattr(assembly_req, 'event_log', []), assembly_start, assembly_end),
        ("Thermal Management", thermal_req, getattr(thermal_req, 'event_log', []), thermal_start, thermal_end)
    ]
    if include_colony:
        phase_schedules.append(("Colony Setup", colony_req, getattr(colony_req, 'event_log', []), colony_start, colony_end))
    for name, req, events, adv_start, adv_end in adv_phase_schedules:
        phase_schedules.append((name, req, events, adv_start, adv_end))

    # 2. Loop by year, process all active phases in parallel
    all_years = set()
    for _, _, _, start, end in phase_schedules:
        all_years.update(range(start, end))
    all_years = sorted(all_years)
    total_time = all_years[-1] - start_year + 1

    # Prepare phase data for output
    plan_data['phases'] = []
    for name, req, events, start, end in phase_schedules:
        phase_data = {
            'name': name,
            'time': req.time_years,
            'mass': req.mass_tons,
            'energy': req.energy_petawatts,
            'risks': req.risks,
            'start_year': start,
            'end_year': end,
            'event_log': events if events else ["No major events occurred during this phase."]
        }
        plan_data['phases'].append(phase_data)

    # 3. Yearly simulation: aggregate events and grid changes
    for idx, year in enumerate(all_years):
        # Determine active phases
        active_phases = [
            (name, req, events, start, end)
            for name, req, events, start, end in phase_schedules
            if start <= year < end
        ]
        # Aggregate events for this year
        year_events = []
        for name, req, events, start, end in active_phases:
            for ev in events:
                if f"Year {year}:" in ev:
                    year_events.append(ev)
        # Progress: use overall progress as fraction of total years
        sim_progress = (year - start_year) / max(1, total_time)
        grid = planner.generate_orbital_grid()
        size = len(grid)
        animated_grid = [[' ' for _ in range(size)] for _ in range(size)]
        center = size // 2
        # 1. Always show resources, hot/cold zones
        for i in range(size):
            for j in range(size):
                cell = grid[i][j]
                if cell in ['R', 'r', 'H', 'C']:
                    animated_grid[i][j] = cell
        # 2. Always show Earth as first habitat (center)
        animated_grid[center][center] = '#'
        # 3. Gradually add collectors in a circular/clockwise pattern
        collector_positions = [(i, j) for i in range(size) for j in range(size) if grid[i][j] == 'O']
        n_collectors = max(1, int(sim_progress * len(collector_positions)))
        def angle_from_center(pos):
            dx, dy = pos[0] - center, pos[1] - center
            return (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
        collector_positions_sorted = sorted(collector_positions, key=angle_from_center)
        active_collectors = collector_positions_sorted[:n_collectors]
        for (i, j) in active_collectors:
            animated_grid[i][j] = 'O'
        # 4. Draw traffic from center to each new collector (clocklike)
        def draw_traffic_line(grid, start, end, symbol='T'):
            x0, y0 = start
            x1, y1 = end
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            if dx > dy:
                err = dx / 2.0
                while x != x1:
                    if grid[x][y] == ' ':
                        grid[x][y] = symbol
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2.0
                while y != y1:
                    if grid[x][y] == ' ':
                        grid[x][y] = symbol
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
            if grid[x][y] == ' ':
                grid[x][y] = symbol
        for (i, j) in active_collectors:
            draw_traffic_line(animated_grid, (center, center), (i, j), symbol='T')
        # 5. Gradually add additional habitats in a circular/clockwise pattern as progress increases
        habitat_positions = [(i, j) for i in range(size) for j in range(size) if grid[i][j] == '#' and (i, j) != (center, center)]
        # Start showing habitats at 20% progress, scale smoothly up to 100%
        if sim_progress > 0.2:
            # Use a smoother scaling function: sqrt(progress) gives a more gradual early growth
            scaled_progress = (sim_progress - 0.2) / 0.8  # Normalize to 0-1 range after 20%
            n_habitats = max(1, int(math.sqrt(scaled_progress) * len(habitat_positions)))
        else:
            n_habitats = 0
        habitat_positions_sorted = sorted(habitat_positions, key=angle_from_center)
        active_habitats = habitat_positions_sorted[:n_habitats]
        for (i, j) in active_habitats:
            animated_grid[i][j] = '#'
        # 6. Event-driven grid modifications (from all year_events)
        event_triggers = {
            # Habitats
            'colony wiped out': lambda grid: grid.__setitem__(center, [' ' for _ in range(size)]),
            'habitat destroyed': lambda grid: grid[center].__setitem__(center, ' '),
            'new habitat established': lambda grid: grid[center][center+1] == ' ' and grid[center].__setitem__(center+1, '#'),
            'habitat expansion': lambda grid: [grid[center].__setitem__(j, '#') for j in range(size) if grid[center][j] == ' '],
            # Collectors
            'collector destroyed': lambda grid: [row.__setitem__(center, ' ') for row in grid],
            'collector malfunction': lambda grid: [row.__setitem__(center, 'X') for row in grid],
            'new collector array completed': lambda grid: [row.__setitem__(center, 'O') for row in grid],
            'collector upgrade': lambda grid: [row.__setitem__(center, 'O') for row in grid],
            # Traffic
            'major traffic accident': lambda grid: [row.__setitem__(center, ' ') for row in grid],
            'traffic breakthrough': lambda grid: [row.__setitem__(center, 'T') for row in grid],
            'traffic rerouted': lambda grid: [row.__setitem__(center, 't') for row in grid],
            # Resources
            'resource depleted': lambda grid: [row.__setitem__(center, ' ') for row in grid],
            'resource discovered': lambda grid: [row.__setitem__(center, 'R') for row in grid],
            # Thermal
            'thermal system failure': lambda grid: [row.__setitem__(center, 'H') for row in grid],
            'thermal breakthrough': lambda grid: [row.__setitem__(center, 'C') for row in grid],
            # Exotic/cosmic
            'quantum anomaly': lambda grid: grid[center][center] == '#' and grid[center].__setitem__(center, 'Q'),
            'black hole event': lambda grid: grid[center][center] == '#' and grid[center].__setitem__(center, 'B'),
            'alien incursion': lambda grid: grid[center][center] == '#' and grid[center].__setitem__(center, 'A'),
            'virtual civilization collapse': lambda grid: grid[center][center] == '#' and grid[center].__setitem__(center, 'V'),
            'singularity event': lambda grid: grid[center][center] == '#' and grid[center].__setitem__(center, 'S'),
        }
        for ev in year_events:
            for trigger, action in event_triggers.items():
                if trigger in ev.lower():
                    action(animated_grid)
        # 7. Final year: show everything
        if sim_progress >= 0.99:
            animated_grid = [row[:] for row in grid]
        frame = {
            'year': year,
            'phase': ', '.join([name for name, _, _, _, _ in active_phases]),
            'progress': sim_progress,
            'grid': animated_grid
        }
        plan_data['visualization']['frames'].append(frame)

    plan_data['visualization']['total_years'] = total_time
    plan_data['visualization']['current_year'] = start_year
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
        'total_energy': total_energy,
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