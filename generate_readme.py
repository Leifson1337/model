# generate_readme.py
import os
import glob
import re

SECTIONS_DIR = "docs/sections"
README_PATH = "README.md"

# Placeholder for dynamic content generation functions
def generate_model_list():
    """
    Placeholder function to generate a list of models.
    In a real implementation, this might scan the 'models/' directory,
    read metadata files, or query a model registry.
    """
    # Example: Scan 'models/' directory for .joblib or .h5 files and their .meta.json
    models_found = []
    if os.path.isdir("models"):
        for item in os.listdir("models"):
            if item.endswith((".joblib", ".h5", ".pkl", ".json", ".ubj", ".txt", ".keras")) and not item.endswith(".meta.json"):
                model_name = os.path.splitext(item)[0]
                # Attempt to find a corresponding .meta.json file
                meta_file = os.path.join("models", f"{model_name}.meta.json")
                if not os.path.exists(meta_file) and item.endswith((".json", ".ubj", ".txt")): # For native model formats
                    meta_file = os.path.join("models", f"{item}.meta.json") # Try exact name match for meta
                
                if os.path.exists(meta_file):
                    models_found.append(f"- {item} (with metadata)")
                else:
                    models_found.append(f"- {item} (metadata missing)")
    if not models_found:
        return "Models directory is empty or no model files found."
    return "\n".join(models_found)

def generate_feature_list():
    """
    Placeholder function to generate a list of features.
    This might read from a configuration file or a dedicated features documentation file.
    """
    # For now, returning a placeholder.
    # In future, this could parse feature engineering scripts or a feature store definition.
    return """
- Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- Rolling Window Features (mean, std)
- Lag Features (price, volume, selected indicators)
- Sentiment Scores (daily aggregated from news)
- Fundamental Data (Revenue, Net Income, Assets, Liabilities, Cash Flow, etc.)
- Target Variable (binary classification based on future price change)
"""

def generate_cli_commands_list():
    """
    Placeholder function to generate CLI command list.
    Ideally, this would parse `python main.py --help` or use Click's context to list commands.
    """
    # For now, a simplified list.
    # Could use subprocess to run `python main.py --help` and parse output.
    return """
    - `python main.py --help`
    - `python main.py load-data --config <config_path>`
    - `python main.py engineer-features --config <config_path>`
    - `python main.py train-model --config <config_path>`
    - `python main.py evaluate --config <config_path>`
    - `python main.py backtest --config <config_path>`
    - `python main.py export --config <config_path>`
    """

def get_current_pipeline_status():
    """
    Placeholder for current pipeline status.
    Could read from a status file, environment variable, or CI/CD system.
    """
    return "Status: In Development (Refactoring and Documentation Phase)"

def get_changelog_summary(num_entries=5):
    """
    Reads the CHANGELOG.md and returns a summary of recent entries.
    """
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # A simple way to get recent entries: find H2 headings (##)
        # This regex looks for "## [Version]" or "## Version"
        # and captures content until the next H2 or end of file.
        entries = re.findall(r"(##\s*\[?[^]]+\]?.*?)(\n##\s*\[?[^]]+\]?|$)", content, re.DOTALL)
        
        summary_lines = []
        for i, (entry_header, _) in enumerate(entries):
            if i < num_entries:
                # Get the header and a few lines of the entry
                entry_content_match = re.search(r"(%s\s*\n((?:-\s*.*?\n)+))" % re.escape(entry_header.strip()), content, re.DOTALL)
                if entry_content_match:
                     summary_lines.append(entry_content_match.group(1).strip())
                else: # Fallback if content parsing is tricky, just use header
                    summary_lines.append(entry_header.strip())
            else:
                break
        
        if not summary_lines:
            return "No changelog entries found or CHANGELOG.md is empty."
            
        return "\n\n".join(summary_lines) + f"\n\n... (see [CHANGELOG.md](./CHANGELOG.md) for full details)"

    except FileNotFoundError:
        return "CHANGELOG.md not found."
    except Exception as e:
        return f"Error reading CHANGELOG.md: {e}"


# Mapping of placeholders to generator functions
DYNAMIC_CONTENT_GENERATORS = {
    "<!-- AUTOGEN:MODEL_LIST -->": generate_model_list,
    "<!-- AUTOGEN:FEATURE_LIST -->": generate_feature_list,
    "<!-- AUTOGEN:CLI_COMMANDS_LIST -->": generate_cli_commands_list,
    "<!-- AUTOGEN:PIPELINE_STATUS -->": get_current_pipeline_status,
    "<!-- AUTOGEN:CHANGELOG_SUMMARY -->": get_changelog_summary,
    # Add more placeholders and their corresponding functions here
}

def main():
    readme_content = []
    section_files = sorted(glob.glob(os.path.join(SECTIONS_DIR, "*.md")))

    if not section_files:
        print(f"No section files found in {SECTIONS_DIR}. README not generated.")
        return

    print(f"Found section files: {section_files}")

    for section_file in section_files:
        try:
            with open(section_file, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Process dynamic content placeholders
                for placeholder, generator_func in DYNAMIC_CONTENT_GENERATORS.items():
                    if placeholder in content:
                        print(f"Generating content for placeholder: {placeholder} in {section_file}")
                        generated_text = generator_func()
                        content = content.replace(placeholder, generated_text)
                
                readme_content.append(content)
        except Exception as e:
            print(f"Error reading or processing file {section_file}: {e}")
            readme_content.append(f"\n\n---\n\n**Error loading section from {os.path.basename(section_file)}**\n\n---\n\n")


    final_readme_text = "\n\n---\n\n".join(readme_content)
    
    # Add a header indicating it's auto-generated
    auto_gen_header = (
        "<!-- README.md is auto-generated from docs/sections/*.md files by generate_readme.py -->\n"
        "<!-- DO NOT EDIT THIS FILE DIRECTLY. Edit the section files or the script. -->\n\n"
    )
    final_readme_text = auto_gen_header + final_readme_text

    try:
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(final_readme_text)
        print(f"Successfully generated {README_PATH}")
    except Exception as e:
        print(f"Error writing {README_PATH}: {e}")

    # Conceptual pre-commit hook information
    pre_commit_hook_info = """
    # Conceptual Pre-commit Hook Setup (e.g., in .pre-commit-config.yaml)
    #
    # repos:
    # -   repo: local
    #     hooks:
    #     -   id: generate-readme
    #         name: Generate README.md
    #         entry: python generate_readme.py
    #         language: system
    #         files: ^docs/sections/.*\.md$|^generate_readme\.py$ # Trigger if section files or script changes
    #         pass_filenames: false # Script handles finding files itself
    #         stages: [commit] # Run on commit
    #
    # To use this:
    # 1. Install pre-commit: `pip install pre-commit`
    # 2. Create `.pre-commit-config.yaml` with the content above.
    # 3. Run `pre-commit install` to set up the git hook.
    # Now, `python generate_readme.py` will run automatically before each commit
    # if relevant files have changed, and the updated README.md will be part of the commit.
    """
    # This could be written to docs/dev_notes.txt or just kept here as a comment.
    # For this task, just having it in the script is fine.
    print("\n--- Pre-commit hook conceptual info ---")
    print(pre_commit_hook_info)


if __name__ == "__main__":
    main()
