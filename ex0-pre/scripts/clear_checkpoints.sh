#!/usr/bin/env bash
# Delete all training checkpoints (asks for confirmation first).
cd "$(dirname "$0")/.."

shopt -s nullglob
files=(checkpoints/*.pt)
if [ ${#files[@]} -eq 0 ]; then
    echo "No checkpoints in checkpoints/"
    exit 0
fi

printf '%s\n' "${files[@]}"
read -r -p "Delete these ${#files[@]} checkpoint(s)? [y/N] " ans
if [[ "$ans" =~ ^[yY] ]]; then
    rm "${files[@]}"
    echo "Deleted."
else
    echo "Aborted."
fi
