#!/bin/bash
# MaxQ Release Script
# Usage: ./tools/release.sh <version>
# Example: ./tools/release.sh 0.1.0

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./tools/release.sh <version>"
    echo "Example: ./tools/release.sh 0.1.0"
    exit 1
fi

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 0.1.0)"
    exit 1
fi

echo "=== Releasing MaxQ v$VERSION ==="
echo

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: Uncommitted changes. Commit or stash before releasing."
    exit 1
fi

# Check we're on main
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "Error: Must release from main branch (currently on $BRANCH)"
    exit 1
fi

# Update version in pyproject.toml
echo "Updating version to $VERSION..."
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Update version in setup.py (if exists)
if [ -f "setup.py" ]; then
    sed -i.bak "s/version=\".*\"/version=\"$VERSION\"/" setup.py
    rm setup.py.bak
fi

# Commit version bump
git add pyproject.toml setup.py 2>/dev/null || git add pyproject.toml
git commit -m "chore: bump version to $VERSION"

# Create tag
git tag -a "v$VERSION" -m "Release v$VERSION"

echo
echo "=== Release v$VERSION created ==="
echo
echo "To publish:"
echo "  git push origin main"
echo "  git push origin v$VERSION"
echo
echo "Or to undo:"
echo "  git tag -d v$VERSION"
echo "  git reset --hard HEAD~1"
echo
