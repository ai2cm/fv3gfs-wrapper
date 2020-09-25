"""
This script takes jinja2 templates in `templates` and converts them into code files in `lib`.
"""
import jinja2
import json
import os
import sys

dim_ranges = {
    "x": "i_start():i_end()",
    "x_interface": "i_start():i_end()+1",
    "y": "j_start():j_end()",
    "y_interface": "j_start():j_end()+1",
    "z": "1:nz()",
    "z_interface": "1:nz()+1",
}

all_templates = (
    "physics_data.F90",
    "_wrapper.pyx",
    "dynamics_data.F90",
    "flagstruct_data.F90",
)
SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
PROPERTIES_DIR = os.path.join(SETUP_DIR, "fv3gfs/wrapper")


def get_dim_range_string(dim_list):
    token_list = [dim_ranges[dim_name] for dim_name in dim_list]
    return ", ".join(reversed(token_list))  # Fortran order is opposite of Python


if __name__ == "__main__":
    requested_templates = sys.argv[1:]

    setup_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(setup_dir, "templates")
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(loader=template_loader, autoescape=True)

    physics_data = json.load(
        open(os.path.join(PROPERTIES_DIR, "physics_properties.json"))
    )
    dynamics_data = json.load(
        open(os.path.join(PROPERTIES_DIR, "dynamics_properties.json"))
    )
    flagstruct_data = json.load(
        open(os.path.join(PROPERTIES_DIR, "flagstruct_properties.json"))
    )

    physics_2d_properties = []
    physics_3d_properties = []
    dynamics_properties = []
    flagstruct_properties = []

    for properties in physics_data:
        if len(properties["dims"]) == 2:
            physics_2d_properties.append(properties)
        elif len(properties["dims"]) == 3:
            physics_3d_properties.append(properties)

    for properties in dynamics_data:
        # properties['first_index'] = ', '.join(0 for item in properties['dims'])
        properties["dim_ranges"] = get_dim_range_string(properties["dims"])
        properties["dim_colons"] = ", ".join(":" for dim in properties["dims"])
        dynamics_properties.append(properties)

    for flag in flagstruct_data:
        if flag["type_fortran"] == "integer":
            flag["type_c"] = "c_int"
            flag["type_cython"] = "int"
        elif flag["type_fortran"] == "real":
            flag["type_c"] = "c_double"
            flag["type_cython"] = "REAL_t"
        elif flag["type_fortran"] == "logical":
            flag["type_c"] = "c_bool"
            flag["type_cython"] = "bint"
        flagstruct_properties.append(flag)

    if len(requested_templates) == 0:
        requested_templates = all_templates

    for base_filename in requested_templates:
        in_filename = os.path.join(setup_dir, f"templates/{base_filename}")
        out_filename = os.path.join(setup_dir, f"lib/{base_filename}")
        template = template_env.get_template(base_filename)
        result = template.render(
            physics_2d_properties=physics_2d_properties,
            physics_3d_properties=physics_3d_properties,
            dynamics_properties=dynamics_properties,
            flagstruct_properties=flagstruct_properties,
        )
        with open(out_filename, "w") as f:
            f.write(result)
