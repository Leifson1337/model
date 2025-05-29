import os
import sys
import click
from pathlib import Path

# Add project root to sys.path to allow importing `main`
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main import cli as main_cli # Import the main CLI group

def get_click_type_name(param_type):
    """Helper to get a user-friendly name for click types."""
    if isinstance(param_type, click.Choice):
        return f"Choice({', '.join(param_type.choices)})"
    if isinstance(param_type, click.Path):
        path_attrs = []
        if param_type.exists: path_attrs.append("exists")
        if param_type.file_okay: path_attrs.append("file_okay")
        if param_type.dir_okay: path_attrs.append("dir_okay")
        if param_type.writable: path_attrs.append("writable")
        if param_type.readable: path_attrs.append("readable")
        if param_type.resolve_path: path_attrs.append("resolve_path")
        return f"Path({', '.join(path_attrs)})"
    return param_type.name.upper() if hasattr(param_type, 'name') else str(param_type)

def format_param_docs(param):
    """Formats documentation for a single Click parameter (option or argument)."""
    docs = ""
    if isinstance(param, click.Option):
        names = "/".join(param.opts)
        docs += f"-   `{names}`"
        if param.metavar:
            docs += f" `{param.metavar}`"
        docs += f" (`{get_click_type_name(param.type)}`)"
        if param.required:
            docs += " [required]"
        if param.help:
            docs += f": {param.help}"
        if param.default is not None and not callable(param.default): # Check for non-callable defaults
            docs += f" (Default: `{param.default}`)"
    elif isinstance(param, click.Argument):
        docs += f"-   `{param.name.upper()}`"
        docs += f" (`{get_click_type_name(param.type)}`)"
        if not param.required: # click.Argument required is True by default
            docs += " [optional]"
        # Arguments don't have help text directly in the same way options do via "help="
        # Their description often comes from the command's docstring.
    return docs

def generate_command_docs(command: click.Command, parent_command_name: str = ""):
    """Generates Markdown documentation for a single Click command."""
    md_lines = []
    full_command_name = f"{parent_command_name} {command.name}".strip()
    
    level = 2 if not parent_command_name else 3 # H2 for top-level, H3 for subcommands
    md_lines.append(f"{'#' * level} Command: `{full_command_name}`")
    md_lines.append("")

    help_text = command.help or command.short_help or "No description provided."
    md_lines.append(help_text)
    md_lines.append("")

    options = [p for p in command.params if isinstance(p, click.Option)]
    arguments = [p for p in command.params if isinstance(p, click.Argument)]

    if options:
        md_lines.append(f"{'#' * (level + 1)} Options")
        md_lines.append("")
        for opt in options:
            md_lines.append(format_param_docs(opt))
        md_lines.append("")

    if arguments:
        md_lines.append(f"{'#' * (level + 1)} Arguments")
        md_lines.append("")
        for arg in arguments:
            md_lines.append(format_param_docs(arg))
        md_lines.append("")
    
    return "\n".join(md_lines)

def generate_all_cli_docs(cli_group: click.Group, command_prefix: str = "python main.py"):
    """Generates Markdown for all commands in a Click group."""
    md_content = ["# CLI Command Reference", ""]
    
    # Document the main group itself if it has help text or options
    # (though typically options are on subcommands)
    # For now, let's assume the main 'cli' group is just a container.

    commands_to_process = [(cli_group, "")] # (command_object, parent_name_str)

    processed_commands = set()

    while commands_to_process:
        current_command, parent_name = commands_to_process.pop(0)
        
        if current_command.name in processed_commands and parent_name == "": # Avoid re-processing top level if aliased
             # This check is basic; complex aliasing or structures might need more robust handling
             pass

        if isinstance(current_command, click.Group):
            # Generate docs for the group itself first
            # Skip if it's the root 'main_cli' group unless it has specific help/options
            if current_command != cli_group or current_command.help:
                 md_content.append(generate_command_docs(current_command, parent_name))
                 md_content.append("\n---\n") # Separator

            # Add subcommands to the queue
            for sub_cmd_name, sub_cmd_obj in sorted(current_command.commands.items()):
                full_parent_name = f"{parent_name} {current_command.name}".strip()
                if sub_cmd_obj not in processed_commands: # Check object, not just name
                    commands_to_process.append((sub_cmd_obj, full_parent_name))
                    processed_commands.add(sub_cmd_obj)
        else: # It's a simple Command
            if current_command not in processed_commands:
                md_content.append(generate_command_docs(current_command, parent_name))
                md_content.append("\n---\n") # Separator
                processed_commands.add(current_command)
                
    return "\n".join(md_content)

if __name__ == "__main__":
    output_dir = project_root / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / "cli.md"

    cli_docs_markdown = generate_all_cli_docs(main_cli)

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(cli_docs_markdown)
        print(f"Successfully generated CLI documentation at: {output_file_path}")
    except IOError as e:
        print(f"Error writing CLI documentation to {output_file_path}: {e}")

    # Example of how to show usage:
    # This would typically be run from the project root.
    # print("\nTo see CLI help directly:")
    # print(f"cd {project_root}")
    # print("python main.py --help")
    # print("python main.py load-data --help") # etc.
