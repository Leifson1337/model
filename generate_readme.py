# generate_readme.py
import os
import glob
import re
import subprocess
import json
import sys
from datetime import datetime # For parsing dates from git log if needed

# Attempt to import GitPython, install if missing
try:
    import git
except ImportError:
    print("GitPython not found. Attempting to install...")
    try:
        # Ensure pip is available and executable
        subprocess.check_call([sys.executable, "-m", "pip", "install", "GitPython"])
        import git
        print("GitPython installed successfully.")
    except subprocess.CalledProcessError as e_pip:
        print(f"Failed to install GitPython using pip: {e_pip}")
        git = None
    except Exception as e_import: # Catch other import errors after attempting install
        print(f"Failed to import GitPython after attempting install: {e_import}")
        git = None

class ReadmeGenerator:
    SECTIONS_DIR = "docs/sections"
    README_PATH = "README.md"
    CHANGELOG_PATH = "CHANGELOG.md"
    MODELS_DIR = "models"
    METRICS_FILE_PATH = "logs/latest_eval_report.json" # Path for latest eval metrics
    NO_DATA_MARKER = object() # Unique marker for no data returned by generator methods

    README_PLACEHOLDER_CONFIG = {
        "{{project_overview}}": {"source_file": "00_overview.md"},
        "{{features_summary}}": {"source_file": "01_features.md"}, # Contains {{feature_list}}
        "{{setup_instructions_link}}": {"content": "[docs/sections/02_setup.md](./docs/sections/02_setup.md) or see [full setup guide](./docs/setup/)"},
        "{{usage_gui_link}}": {"content": "[docs/sections/03_usage_gui.md](./docs/sections/03_usage_gui.md)"},
        "{{cli_commands_summary}}": {
            "method_name": "_generate_cli_commands_summary",
            "render_if_empty": True, # Default, but explicit
            "empty_message": "CLI commands could not be determined. Run `python main.py --help` manually."
        },
        "{{api_usage_link}}": {"content": "[docs/sections/05_usage_api.md](./docs/sections/05_usage_api.md)"},
        "{{examples_link}}": {"content": "[docs/sections/06_examples.md](./docs/sections/06_examples.md)"}, # Contains {{examples_list}}
        "{{architecture_overview_link}}": {"source_file": "07_architecture.md"}, # Contains {{models_overview}}
        "{{models_overview}}": {
            "method_name": "_generate_models_overview",
            "render_if_empty": True,
            "empty_message": "No models with metadata (`.meta.json` files) found in `models/` directory. Train some models first!"
        },
        "{{latest_evaluation_metrics}}": {
            "method_name": "_generate_latest_evaluation_metrics",
            "render_if_empty": False, # Hide this section if no metrics file
            "empty_message": "" # Not strictly needed if render_if_empty is False
        },
        "{{changelog_summary}}": {
            "method_name": "_get_changelog_summary",
            "render_if_empty": True,
            "empty_message": "No changelog information available (no Git history or CHANGELOG.md found)."
        },
        "{{roadmap_link}}": {"source_file": "08_roadmap.md"}, # Contains {{pipeline_status}}
        "{{pipeline_status}}": {"method_name": "_get_current_pipeline_status"}, # Assumed to always return something
        "{{contributing_guidelines}}": {"source_file": "10_contributing.md"},
        "{{license_info}}": {"source_file": "11_license.md"},
        # Internal placeholders (processed when their containing section file is loaded)
        "{{feature_list}}": {"method_name": "_generate_feature_list"}, # Assumed to always return content
        "{{examples_list}}": {
            "method_name": "_generate_examples_list",
            "render_if_empty": True, # Example for an internal placeholder
            "empty_message": "No examples documented yet."
            },
    }

    def __init__(self, template_path="README_TEMPLATE.md"):
        self.template_path = template_path

    def _generate_models_overview(self):
        if not os.path.isdir(self.MODELS_DIR):
            return self.NO_DATA_MARKER
        meta_files = glob.glob(os.path.join(self.MODELS_DIR, "*.meta.json"))
        if not meta_files:
            return self.NO_DATA_MARKER

        table_header = "| Model File | Timestamp (UTC) | Key Metric(s) | Features Info |\n"
        table_separator = "|--------------|-------------------|-----------------|---------------|\n"
        table_rows = []

        for meta_file_path in meta_files:
            try:
                with open(meta_file_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                model_file = meta_data.get("model_name", os.path.basename(meta_file_path).replace(".meta.json", ""))
                timestamp = meta_data.get("timestamp_utc", meta_data.get("timestamp", "N/A"))
                
                metrics_str = "N/A"
                metrics_data = meta_data.get("metrics", {})
                if metrics_data:
                    if "accuracy" in metrics_data: metrics_str = f"Acc: {metrics_data['accuracy']:.3f}"
                    elif "f1_score" in metrics_data: metrics_str = f"F1: {metrics_data['f1_score']:.3f}"
                    elif "auc" in metrics_data: metrics_str = f"AUC: {metrics_data['auc']:.3f}"
                    else: key, val = next(iter(metrics_data.items())); metrics_str = f"{key[:4]}: {val:.3f}" if isinstance(val, float) else f"{key[:4]}: {val}"

                features_str = "N/A"
                feature_config = meta_data.get("feature_config_used", meta_data.get("feature_config", {}))
                if feature_config:
                    features_used = feature_config.get("features_used")
                    if features_used and isinstance(features_used, list):
                        features_str = f"{len(features_used)} (e.g., {', '.join(features_used[:2])}...)"
                    elif isinstance(feature_config, dict) and feature_config:
                        features_str = "Configured (see meta)"
                
                table_rows.append(f"| `{model_file}` | {timestamp} | {metrics_str} | {features_str} |")
            except Exception as e:
                print(f"Warning: Error processing metadata file {meta_file_path}: {e}")
                table_rows.append(f"| {os.path.basename(meta_file_path).replace('.meta.json','')} | N/A | Error parsing | N/A |")
        
        return table_header + table_separator + "\n".join(table_rows) if table_rows else self.NO_DATA_MARKER

    def _generate_feature_list(self):
        return ("- Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)\n"
                "- Rolling Window Features (mean, std)\n"
                "- Lag Features (price, volume, selected indicators)\n"
                "- Sentiment Scores (daily aggregated from news)\n"
                "- Fundamental Data (Revenue, Net Income, Assets, Liabilities, Cash Flow, etc.)\n"
                "- Target Variable (binary classification based on future price change)")

    def _generate_cli_commands_summary(self):
        try:
            python_executable = sys.executable
            result = subprocess.run([python_executable, "main.py", "--help"], capture_output=True, text=True, check=True, timeout=10)
            help_text = result.stdout
            commands_section_match = re.search(r"Commands:\s*\n([\s\S]*)", help_text)
            if not commands_section_match:
                return self.NO_DATA_MARKER

            commands_text = commands_section_match.group(1).strip()
            command_lines = commands_text.split('\n')
            
            markdown_list = ["**Key CLI Commands:**\n"]
            for line in command_lines:
                line = line.strip()
                if not line: continue
                match = re.match(r"^\s*([a-zA-Z0-9_-]+)\s+(.*)", line)
                if match:
                    command, description = match.group(1), match.group(2).strip()
                    markdown_list.append(f"-   `{command}`: {description}")
            
            return "\n".join(markdown_list) + "\n\nRun `python main.py <command> --help` for more details." if len(markdown_list) > 1 else self.NO_DATA_MARKER
        except FileNotFoundError: return self.NO_DATA_MARKER
        except subprocess.TimeoutExpired: return self.NO_DATA_MARKER
        except subprocess.CalledProcessError as e: return self.NO_DATA_MARKER
        except Exception as e: return self.NO_DATA_MARKER

    def _generate_examples_list(self):
        return "(More examples will be listed here as they are documented in `docs/workflows/`.)"
    
    def _generate_latest_evaluation_metrics(self):
        try:
            with open(self.METRICS_FILE_PATH, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            has_model_name = report.get("model_name")
            has_metrics = report.get("metrics")
            # Ensure metrics is a non-empty dict if it exists
            metrics_content_present = isinstance(has_metrics, dict) and bool(has_metrics)
            
            print(f"Debug _generate_latest_evaluation_metrics: model_name found: {bool(has_model_name)}, metrics_content_present: {metrics_content_present}")

            if not has_model_name and not metrics_content_present:
                 return self.NO_DATA_MARKER

            metrics_md = [
                f"**Latest Evaluation Report ({report.get('report_generated_at', 'N/A')})**",
                f"- **Model:** `{report.get('model_name', 'N/A')}`",
                f"- **Dataset:** `{report.get('dataset_used', 'N/A')}`",
                f"- **Type:** {report.get('evaluation_type', 'N/A')}",
                "**Key Metrics:**"
            ]
            if metrics_content_present:
                for key, value in has_metrics.items(): # Use has_metrics which is the dict
                    metrics_md.append(f"  - {key.replace('_', ' ').title()}: {value}")
            else:
                metrics_md.append("  - No metrics data available.")
            if report.get("notes"): metrics_md.append(f"\n**Notes:** {report['notes']}")
            return "\n".join(metrics_md)
        except FileNotFoundError: return self.NO_DATA_MARKER
        except json.JSONDecodeError: return self.NO_DATA_MARKER
        except Exception as e: return self.NO_DATA_MARKER

    def _get_current_pipeline_status(self):
        return "Status: In Development (Refactoring and Documentation Phase)"

    def _get_changelog_summary(self, num_entries=7):
        global git
        changelog_entries = []
        try:
            if git:
                repo = git.Repo(search_parent_directories=True)
                commits = list(repo.iter_commits(max_count=num_entries * 2))
                for commit in commits:
                    if len(changelog_entries) >= num_entries: break
                    msg_first_line = commit.message.split('\n')[0]
                    try:
                        commit_date_str = commit.authored_datetime.strftime('%Y-%m-%d')
                    except Exception: 
                        commit_date_str = "N/A"
                    entry = f"- `{commit.hexsha[:7]}`: {msg_first_line} ({commit_date_str} by {commit.author.name})"
                    if re.match(r"^(feat|fix|docs|refactor|perf|ci|build|style|test)\b", msg_first_line, re.IGNORECASE):
                        changelog_entries.append(entry)
                    elif len(changelog_entries) < num_entries / 2: 
                         changelog_entries.append(entry)
                if not changelog_entries and commits: 
                    changelog_entries = [f"- `{c.hexsha[:7]}`: {c.message.splitlines()[0]} ({c.authored_datetime.strftime('%Y-%m-%d')} by {c.author.name})" for c in commits[:num_entries]]
            else: 
                cmd = ['git', 'log', f'--pretty=format:- `%h` %s (%ad by %an)', '--date=short', f'-n{num_entries}']
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
                changelog_entries = result.stdout.strip().split('\n')

            if changelog_entries:
                return "**Recent Changes (from Git History):**\n" + "\n".join(changelog_entries) + \
                       "\n\n... (see full commit history for more details)"
        except (git.InvalidGitRepositoryError, FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Git-based changelog failed: {e}. Falling back to {self.CHANGELOG_PATH} if available.")
        except Exception as e: 
            print(f"An unexpected error occurred with GitPython: {e}. Falling back to {self.CHANGELOG_PATH}.")
        
        try:
            with open(self.CHANGELOG_PATH, "r", encoding="utf-8") as f: content = f.read()
            entries = re.findall(r"(##\s*\[?[^]]+\]?.*?)(\n##\s*\[?[^]]+\]?|$)", content, re.DOTALL)
            summary_lines = [entries[i][0].strip() for i in range(min(len(entries), num_entries))]
            if not summary_lines: return self.NO_DATA_MARKER
            return "\n\n".join(summary_lines) + f"\n\n... (see [{self.CHANGELOG_PATH}](./{self.CHANGELOG_PATH}) for full details)"
        except FileNotFoundError: return self.NO_DATA_MARKER
        except Exception as e_file: return self.NO_DATA_MARKER

    def generate(self):
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                template_content = f.read()
        except FileNotFoundError:
            print(f"Error: Template file '{self.template_path}' not found. README not generated.")
            return
        except Exception as e:
            print(f"Error reading template file '{self.template_path}': {e}")
            return

        processed_content = template_content
        for placeholder, config_entry in self.README_PLACEHOLDER_CONFIG.items():
            generated_text = ""
            is_dynamic_method = "method_name" in config_entry
            generated_data_for_check = None # Store result of method call if dynamic
            
            if is_dynamic_method:
                method = getattr(self, config_entry["method_name"], None)
                if method:
                    generated_data_for_check = method() 
                    
                    if generated_data_for_check == self.NO_DATA_MARKER:
                        render_if_empty = config_entry.get("render_if_empty", True)
                        if render_if_empty:
                            generated_text = config_entry.get("empty_message", "")
                        else:
                            placeholder_line_regex = r"^[ \t]*" + re.escape(placeholder) + r"[ \t]*\n?"
                            if re.search(placeholder_line_regex, processed_content, re.MULTILINE):
                                processed_content = re.sub(placeholder_line_regex, "", processed_content, flags=re.MULTILINE)
                                print(f"Hid section for empty placeholder: {placeholder}")
                            else: 
                                generated_text = "" 
                            continue 
                    else: 
                        generated_text = generated_data_for_check or "" 
                else: 
                    generated_text = f"[[Method {config_entry['method_name']} not found for {placeholder}]]"
            
            elif "source_file" in config_entry:
                try:
                    with open(os.path.join(self.SECTIONS_DIR, config_entry["source_file"]), "r", encoding="utf-8") as f:
                        generated_text = f.read()
                    generated_text = self._process_internal_placeholders(generated_text)
                except FileNotFoundError: generated_text = f"[[Source file {config_entry['source_file']} not found for {placeholder}]]"
                except Exception as e: generated_text = f"[[Error loading {config_entry['source_file']} for {placeholder}: {e}]]"
            
            elif "content" in config_entry: 
                 generated_text = config_entry["content"]

            if not (is_dynamic_method and generated_data_for_check == self.NO_DATA_MARKER and not config_entry.get("render_if_empty", True)) :
                 processed_content = processed_content.replace(placeholder, generated_text)
            
        for ph in re.findall(r"\{\{[a-zA-Z0-9_]+\}\}", processed_content):
            print(f"Warning: Unprocessed placeholder '{ph}' found in final content.")

        try:
            with open(self.README_PATH, "w", encoding="utf-8") as f:
                f.write(processed_content)
            print(f"Successfully generated {self.README_PATH} from {self.template_path}")
        except Exception as e:
            print(f"Error writing {self.README_PATH}: {e}")

    def _process_internal_placeholders(self, content_str: str) -> str:
        processed_content = content_str
        for placeholder, config_entry in self.README_PLACEHOLDER_CONFIG.items():
            if placeholder in processed_content and "method_name" in config_entry:
                method = getattr(self, config_entry["method_name"], None)
                if method:
                    generated_data = method()
                    generated_text_internal = "" 

                    if generated_data == self.NO_DATA_MARKER:
                        render_if_empty_internal = config_entry.get("render_if_empty", True)
                        if render_if_empty_internal:
                            generated_text_internal = config_entry.get("empty_message", "")
                        else:
                            generated_text_internal = "" 
                    else:
                        generated_text_internal = generated_data or ""
                    
                    processed_content = processed_content.replace(placeholder, generated_text_internal)
        
        for old_placeholder, generator_method_name_str in self._get_old_dynamic_generators().items():
            if old_placeholder in processed_content:
                generator_method = getattr(self, generator_method_name_str, None)
                if generator_method:
                    generated_text = generator_method() or ""
                    processed_content = processed_content.replace(old_placeholder, generated_text)
        return processed_content

    def _get_old_dynamic_generators(self):
        return {
            "<!-- AUTOGEN:MODEL_LIST -->": "_generate_models_overview",
            "<!-- AUTOGEN:FEATURE_LIST -->": "_generate_feature_list",
            "<!-- AUTOGEN:CLI_COMMANDS_LIST -->": "_generate_cli_commands_summary",
            "<!-- AUTOGEN:PIPELINE_STATUS -->": "_get_current_pipeline_status",
            "<!-- AUTOGEN:CHANGELOG_SUMMARY -->": "_get_changelog_summary",
            "<!-- AUTOGEN:EXAMPLES_LIST -->": "_generate_examples_list",
        }
        
    @staticmethod
    def print_pre_commit_hook_info():
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
    #         files: ^README_TEMPLATE\.md$|^docs/sections/.*\.md$|^generate_readme\.py$|^CHANGELOG.md$|^logs/latest_eval_report.json$|^models/.*\.meta\.json$
    #         pass_filenames: false 
    #         stages: [commit] 
    #
    # To use this:
    # 1. Install pre-commit: `pip install pre-commit`
    # 2. Create `.pre-commit-config.yaml` with the content above.
    # 3. Run `pre-commit install` to set up the git hook.
    # Now, this script will run automatically before each commit if relevant files change.
    """
        print("\n--- Pre-commit hook conceptual info ---")
        print(pre_commit_hook_info)

if __name__ == "__main__":
    generator = ReadmeGenerator()
    generator.generate()
    generator.print_pre_commit_hook_info()
