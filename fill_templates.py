"""
This script takes jinja2 templates in `templates` and converts them into code files in `lib`.
"""
import jinja2
import json
import os

dim_ranges = {
    'x': 'i_start():i_end()',
    'x_interface': 'i_start():i_end()+1',
    'y': 'j_start():j_end()',
    'y_interface': 'j_start():j_end()+1',
    'z': 'nz()',
}


def get_dim_range_string(dim_list):
    token_list = [dim_ranges[dim_name] for dim_name in dim_list]
    return ', '.join(reversed(token_list))  # Fortran order is opposite of Python


if __name__ == '__main__':
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(setup_dir, 'templates')
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(loader=template_loader, autoescape=True)

    physics_data = json.load(open(os.path.join(setup_dir, 'fv3gfs/physics_properties.json')))
    dynamics_data = json.load(open(os.path.join(setup_dir, 'fv3gfs/dynamics_properties.json')))

    physics_2d_properties = []
    physics_3d_properties = []
    dynamics_properties = []

    for properties in physics_data:
        if len(properties['dims']) == 2:
            physics_2d_properties.append(properties)
        elif len(properties['dims']) == 3:
            physics_3d_properties.append(properties)

    for properties in dynamics_data:
        # properties['first_index'] = ', '.join(0 for item in properties['dims'])
        properties['dim_ranges'] = get_dim_range_string(properties['dims'])
        properties['dim_colons'] = ', '.join(':' for dim in properties['dims'])
        dynamics_properties.append(properties)

    for in_filename in ('physics_data.F90', '_wrapper.pyx', 'dynamics_data.F90'):
        template = template_env.get_template(in_filename)
        result = template.render(
            physics_2d_properties=physics_2d_properties, physics_3d_properties=physics_3d_properties,
            dynamics_properties=dynamics_properties,
        )
        with open(os.path.join(setup_dir, f'lib/{in_filename}'), 'w') as f:
            f.write(result)
