import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Optional # Added import for Optional

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
UNRELEASED_SECTION_HEADER = "## [Unreleased]"

def get_git_output(command: list[str]) -> str:
    """Runs a git command and returns its stdout."""
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return process.stdout.strip()
    except FileNotFoundError:
        print("Error: Git command not found. Is Git installed and in PATH?")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"Error running git command '{' '.join(command)}': {e.stderr}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred with git command '{' '.join(command)}': {e}")
        return ""

def get_tags_sorted_by_date() -> list[tuple[str, str]]:
    """Returns a list of (tag_name, tag_date_iso) sorted by date (latest first)."""
    raw_tags_output = get_git_output(['git', 'tag', '--sort=-committerdate', '--format=%(refname:short) %(committerdate:iso)'])
    if not raw_tags_output:
        return []
    
    tags_with_dates = []
    for line in raw_tags_output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            tag_name = parts[0]
            try:
                # Parse the full ISO date and format to YYYY-MM-DD
                # Example committerdate:iso format: 2023-10-26 10:30:00 -0400
                # Need to handle potential spaces in timezone offset if not using strict iso parsing
                # For simplicity, assuming the date part is the first part after tag name
                date_str_part = parts[1] # YYYY-MM-DD
                datetime.strptime(date_str_part, '%Y-%m-%d') # Validate date format
                tags_with_dates.append((tag_name, date_str_part))
            except ValueError:
                # Try to parse more complex ISO format if simple YYYY-MM-DD fails
                try:
                    full_date_str = " ".join(parts[1:]) # Rejoin if date/time/tz was split
                    tag_date_obj = datetime.fromisoformat(full_date_str.replace(" ", "", 1)) # fromisoformat expects 'YYYY-MM-DDTHH:MM:SS[+/-HH:MM]' or similar
                    tag_date_str = tag_date_obj.strftime('%Y-%m-%d')
                    tags_with_dates.append((tag_name, tag_date_str))
                except ValueError:
                    print(f"Warning: Could not parse date for tag {tag_name}: {' '.join(parts[1:])}")
                    tags_with_dates.append((tag_name, "N/A")) # Fallback
    return tags_with_dates


def get_commits_between(from_ref: Optional[str], to_ref: str) -> list[str]:
    """Gets commit messages between two references."""
    range_spec = f"{from_ref}..{to_ref}" if from_ref else to_ref
    log_format = "--pretty=format:- %s (%h by %an, %ad)"
    date_format = "--date=short" # YYYY-MM-DD
    
    commits_output = get_git_output(['git', 'log', range_spec, log_format, date_format])
    if not commits_output:
        # Provide more context based on whether it's an unreleased section or a tagged release
        if to_ref == "HEAD" and from_ref: # Unreleased section for an existing project
            return ["  - No new changes since the last release."]
        elif not from_ref: # Initial set of commits before the first tag, or empty repo
             return ["  - No commits found in this range (e.g. initial commits before first tag, or empty history)."]
        return ["  - No specific commits found for this version range (possibly a direct tag on a commit already listed)."]
        
    return commits_output.splitlines()

def generate_changelog_content() -> list[str]:
    """Generates the content for CHANGELOG.md."""
    lines = ["# Changelog", "", "All notable changes to this project will be documented in this file.", ""]
    
    tags = get_tags_sorted_by_date()
    
    latest_tag_name = tags[0][0] if tags else None
    lines.append(UNRELEASED_SECTION_HEADER)
    unreleased_commits = get_commits_between(latest_tag_name, "HEAD")
    lines.extend(unreleased_commits)
    lines.append("")

    for i, (tag_name, tag_date) in enumerate(tags):
        previous_tag_name = tags[i+1][0] if (i + 1) < len(tags) else None
        lines.append(f"## [{tag_name}] - {tag_date}")
        commits_for_tag = get_commits_between(previous_tag_name, tag_name)
        lines.extend(commits_for_tag)
        lines.append("")
        
    if not tags and len(unreleased_commits) == 1 and "No new changes" in unreleased_commits[0]:
        # If no tags and no actual unreleased changes, provide a general message
        lines.append("  - No significant changes or tags found in the project history yet.")

    return lines

if __name__ == "__main__":
    changelog_lines = generate_changelog_content()
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(changelog_lines))
        print(f"Successfully generated CHANGELOG.md at {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing CHANGELOG.md to {OUTPUT_FILE}: {e}")
