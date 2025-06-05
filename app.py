from flask import Flask, render_template, request, jsonify
from dyson_sphere_planner import DysonSpherePlanner, DysonStrategy
import json

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
    
    planner = DysonSpherePlanner(strategy, include_colony, start_year)
    
    # Generate the plan and collect all output
    plan_data = {
        'phases': [],
        'metrics': {},
        'grid': [],
        'warnings': []
    }
    
    # Capture the plan generation
    phases = [
        ("Material Extraction", planner.simulate_material_extraction()),
        ("Transport System", planner.simulate_transport_system()),
        ("Assembly", planner.simulate_assembly()),
        ("Thermal Management", planner.simulate_thermal_management())
    ]
    
    if strategy in [DysonStrategy.COSMIC, DysonStrategy.DIMENSIONAL, 
                   DysonStrategy.SINGULARITY, DysonStrategy.OMNIVERSAL]:
        phases.extend([
            ("Cosmic Engineering", planner.simulate_cosmic_engineering()),
            ("Dimensional Engineering", planner.simulate_dimensional_engineering()),
            ("Singularity Evolution", planner.simulate_singularity_evolution())
        ])
    
    if include_colony:
        phases.append(("Colony Setup", planner.simulate_colony_setup()))
    
    # Process phases
    total_time = 0
    for phase_name, requirements in phases:
        phase_data = {
            'name': phase_name,
            'time': requirements.time_years,
            'mass': requirements.mass_tons,
            'energy': requirements.energy_petawatts,
            'risks': requirements.risks
        }
        plan_data['phases'].append(phase_data)
        total_time += requirements.time_years
    
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
        plan_data['metrics'].update({
            'pocket_universes': planner.quantum_engineer.cosmic_engineer.universes_created,
            'black_holes': planner.quantum_engineer.cosmic_engineer.black_holes_harvested,
            'dimensions': len(planner.quantum_engineer.dimensional_engineer.dimensions_accessed),
            'consciousness_uploads': planner.quantum_engineer.singularity_evolution.consciousness_uploads,
            'ai_evolution': planner.quantum_engineer.singularity_evolution.ai_evolution_level,
            'digital_realities': len(planner.quantum_engineer.singularity_evolution.digital_realities)
        })
    
    # Generate orbital grid
    grid = planner.generate_orbital_grid()
    plan_data['grid'] = grid
    
    return jsonify(plan_data)

if __name__ == '__main__':
    app.run(debug=True, port=5005) 