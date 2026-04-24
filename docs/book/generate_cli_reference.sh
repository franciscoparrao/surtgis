#!/bin/bash
# Generate docs/book/src/reference/cli/*.md from `surtgis <cmd> --help`.
#
# Covers the full two-level CLI surface: each top-level command gets one
# markdown page. For commands with subcommands (terrain, hydrology, stac,
# etc.), we emit a section per subcommand inside the same page rather than
# exploding into many files — keeps navigation usable.
#
# Re-run whenever the CLI surface changes. Intended to be dumb: the source of
# truth is `--help`, not hand-edited prose.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
OUT="$ROOT/docs/book/src/reference/cli"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first: cargo build --release -p surtgis"
    exit 1
fi

mkdir -p "$OUT"

# Fence a help block with language hint so syntax highlighter picks up flags.
fence_help() {
    local text="$1"
    printf '```text\n%s\n```\n' "$text"
}

# Strip ANSI escape codes that clap emits for terminals — makes the markdown
# safe to render on any theme.
strip_ansi() {
    sed -r 's/\x1B\[[0-9;]*[a-zA-Z]//g'
}

# Top-level commands to document. Excludes `help` (meta) and aliases.
TOP_COMMANDS=(
    info
    terrain
    hydrology
    imagery
    morphology
    landscape
    extract
    extract-patches
    clip
    rasterize
    resample
    mosaic
    cog
    stac
    pipeline
    vector
    interpolation
    temporal
    classification
    texture
    statistics
)

# --- Write index page ---
{
    echo "# CLI reference"
    echo ""
    echo "Every top-level \`surtgis\` subcommand, with flags and a minimal example."
    echo "Pages are generated from \`--help\` output and regenerated on every"
    echo "CLI-surface change."
    echo ""
    echo "## Top-level commands"
    echo ""
    for cmd in "${TOP_COMMANDS[@]}"; do
        desc=$("$BIN" --help 2>&1 | strip_ansi | awk -v cmd="$cmd" '
            /^Commands:/ { in_cmds=1; next }
            in_cmds && /^Options:/ { exit }
            in_cmds {
                # match lines like "  cmd    Description"
                line = $0
                sub(/^[[:space:]]+/, "", line)
                n = split(line, parts, /[[:space:]][[:space:]]+/)
                if (n >= 2 && parts[1] == cmd) {
                    print parts[2]
                    exit
                }
            }
        ')
        echo "- [\`surtgis $cmd\`](./$cmd.md) — $desc"
    done
    echo ""
    echo "## Global flags"
    echo ""
    echo "Available on every subcommand:"
    echo ""
    "$BIN" --help 2>&1 | strip_ansi | awk '
        /^Options:/ { in_opts=1; next }
        in_opts && /^$/ { exit }
        in_opts { print }
    ' | fence_help "$(cat)" > /tmp/_surtgis_globals.md
    cat /tmp/_surtgis_globals.md
    rm -f /tmp/_surtgis_globals.md
} > "$OUT/index.md"

# --- Write one page per top-level command ---
for cmd in "${TOP_COMMANDS[@]}"; do
    page="$OUT/$cmd.md"
    help_text=$("$BIN" "$cmd" --help 2>&1 | strip_ansi)

    # Does this command have subcommands?
    subs=$(echo "$help_text" | awk '
        /^Commands:/ { in_cmds=1; next }
        in_cmds && /^Options:/ { exit }
        in_cmds && /^  [a-z]/ {
            sub(/^[[:space:]]+/, "")
            split($0, parts, /[[:space:]]/)
            if (parts[1] != "help") print parts[1]
        }
    ')

    {
        echo "# \`surtgis $cmd\`"
        echo ""
        # Extract one-line summary from top of help
        summary=$(echo "$help_text" | head -1)
        if [ -n "$summary" ]; then
            echo "_${summary}_"
            echo ""
        fi

        if [ -z "$subs" ]; then
            # Leaf command — just dump --help
            echo "## Synopsis"
            echo ""
            printf '```text\n%s\n```\n' "$help_text"
        else
            # Command with subcommands — emit index + one section per sub
            echo "## Overview"
            echo ""
            printf '```text\n%s\n```\n\n' "$help_text"
            for sub in $subs; do
                sub_help=$("$BIN" "$cmd" "$sub" --help 2>&1 | strip_ansi || echo "(help unavailable)")
                anchor=$(echo "$sub" | tr '[:upper:]_' '[:lower:]-')
                echo "## \`$cmd $sub\` {#$anchor}"
                echo ""
                printf '```text\n%s\n```\n\n' "$sub_help"
            done
        fi
    } > "$page"
done

echo "Wrote $(ls "$OUT"/*.md | wc -l) files to $OUT"
