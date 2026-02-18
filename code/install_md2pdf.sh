#!/bin/bash
# Segment Platform — Dependency installer for md2pdf.sh
# Installs all required packages for Markdown to PDF conversion
#
# Usage:
#   ./code/install_md2pdf.sh
#   ./code/install_md2pdf.sh --verbose

set -e

# ── Configuration ───────────────────────────────────────────────────────────────
VERBOSE=false
for arg in "$@"; do
    case "$arg" in
        --verbose) VERBOSE=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Detect package manager ───────────────────────────────────────────────────────
if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt-get update && sudo apt-get install -y"
    PKG_LIST="pandoc texlive-xetex texlive-latex-extra texlive-fonts-recommended librsvg2-bin libfontconfig1-dev fontconfig"
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    INSTALL_CMD="sudo yum install -y"
    PKG_LIST="pandoc texlive-xetex texlive-collection-latexextra librsvg2-devel fontconfig-devel"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
    PKG_LIST="pandoc texlive-xetex texlive-collection-latexextra librsvg2-devel fontconfig-devel"
elif command -v pacman >/dev/null 2>&1; then
    PKG_MANAGER="pacman"
    INSTALL_CMD="sudo pacman -S --noconfirm"
    PKG_LIST="pandoc texlive-xetex texlive-latexextra librsvg libfontconfig"
elif command -v brew >/dev/null 2>&1; then
    PKG_MANAGER="brew"
    INSTALL_CMD="brew install"
    PKG_LIST="pandoc mactex-no-gui librsvg fontconfig"
else
    echo "Error: No supported package manager found (apt, yum, dnf, pacman, brew)"
    exit 1
fi

$VERBOSE && echo "Detected package manager: $PKG_MANAGER"
$VERBOSE && echo "Install command: $INSTALL_CMD"
$VERBOSE && echo "Package list: $PKG_LIST"

# ── Check what's already installed ───────────────────────────────────────────────────
echo "Checking installed packages..."

# Check pandoc
if command -v pandoc >/dev/null 2>&1; then
    PANDOC_VERSION=$(pandoc --version | head -n1)
    echo "✓ pandoc already installed: $PANDOC_VERSION"
    PANDOC_INSTALLED=true
else
    echo "✗ pandoc not found"
    PANDOC_INSTALLED=false
fi

# Check xelatex
if command -v xelatex >/dev/null 2>&1; then
    XELATEX_VERSION=$(xelatex --version | head -n1)
    echo "✓ xelatex already installed: $XELATEX_VERSION"
    XELATEX_INSTALLED=true
else
    echo "✗ xelatex not found"
    XELATEX_INSTALLED=false
fi

# Check pandoc-crossref
if command -v pandoc-crossref >/dev/null 2>&1; then
    CROSSREF_VERSION=$(pandoc-crossref --version | head -n1)
    echo "✓ pandoc-crossref already installed: $CROSSREF_VERSION"
    CROSSREF_INSTALLED=true
else
    echo "✗ pandoc-crossref not found"
    CROSSREF_INSTALLED=false
fi

# ── Install missing packages ───────────────────────────────────────────────────────
if [ "$PANDOC_INSTALLED" = false ] || [ "$XELATEX_INSTALLED" = false ]; then
    echo ""
    echo "Installing missing packages..."
    $VERBOSE && echo "Running: $INSTALL_CMD $PKG_LIST"
    
    if [ "$PKG_MANAGER" = "brew" ] && [ "$XELATEX_INSTALLED" = false ]; then
        echo "Note: Installing MacTeX (this may take a while and is large)..."
        echo "You may be prompted for your password multiple times."
    fi
    
    eval $INSTALL_CMD $PKG_LIST
    echo "✓ System packages installed"
else
    echo "✓ All system packages already installed"
fi

# ── Install pandoc-crossref (if needed) ─────────────────────────────────────────────
if [ "$CROSSREF_INSTALLED" = false ]; then
    echo ""
    echo "Installing pandoc-crossref..."
    
    # Try different installation methods
    if command -v cargo >/dev/null 2>&1; then
        echo "Installing via cargo..."
        cargo install pandoc-crossref
    elif command -v wget >/dev/null 2>&1; then
        echo "Downloading pre-built binary..."
        # Detect architecture
        ARCH=$(uname -m)
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        
        if [ "$OS" = "linux" ]; then
            if [ "$ARCH" = "x86_64" ]; then
                BINARY_URL="https://github.com/lierdakil/pandoc-crossref/releases/latest/download/pandoc-crossref-linux"
            elif [ "$ARCH" = "aarch64" ]; then
                BINARY_URL="https://github.com/lierdakil/pandoc-crossref/releases/latest/download/pandoc-crossref-linux-arm64"
            else
                echo "Unsupported architecture: $ARCH"
                echo "Please install pandoc-crossref manually or install Rust to build from source"
                exit 1
            fi
        elif [ "$OS" = "darwin" ]; then
            if [ "$ARCH" = "x86_64" ]; then
                BINARY_URL="https://github.com/lierdakil/pandoc-crossref/releases/latest/download/pandoc-crossref-mac"
            elif [ "$ARCH" = "arm64" ]; then
                BINARY_URL="https://github.com/lierdakil/pandoc-crossref/releases/latest/download/pandoc-crossref-mac-arm64"
            else
                echo "Unsupported architecture: $ARCH"
                echo "Please install pandoc-crossref manually"
                exit 1
            fi
        else
            echo "Unsupported OS: $OS"
            echo "Please install pandoc-crossref manually"
            exit 1
        fi
        
        wget -O /tmp/pandoc-crossref "$BINARY_URL"
        chmod +x /tmp/pandoc-crossref
        sudo mv /tmp/pandoc-crossref /usr/local/bin/
        echo "✓ pandoc-crossref installed"
    else
        echo "Error: Neither cargo nor wget found"
        echo "Please install pandoc-crossref manually:"
        echo "  - With cargo: cargo install pandoc-crossref"
        echo "  - Or download from: https://github.com/lierdakil/pandoc-crossref/releases"
        exit 1
    fi
else
    echo "✓ pandoc-crossref already installed"
fi

# ── Install fonts (optional but recommended) ─────────────────────────────────────────
echo ""
echo "Checking fonts..."

# Function to check if font is available
check_font() {
    fc-list | grep -qi "$1"
}

# Function to install font on Linux
install_font_linux() {
    FONT_NAME="$1"
    FONT_URL="$2"
    FONT_DIR="$3"
    
    if check_font "$FONT_NAME"; then
        echo "✓ $FONT_NAME already available"
        return 0
    fi
    
    echo "Installing $FONT_NAME..."
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    if command -v wget >/dev/null 2>&1; then
        wget -q "$FONT_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -sLO "$FONT_URL"
    else
        echo "✗ Cannot download font (neither wget nor curl available)"
        cd - >/dev/null
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    sudo mkdir -p "$FONT_DIR"
    sudo unzip -q "*.zip" -d "$FONT_DIR" 2>/dev/null || sudo mv *.ttf *.otf "$FONT_DIR" 2>/dev/null || true
    sudo fc-cache -f
    cd - >/dev/null
    rm -rf "$TEMP_DIR"
    
    if check_font "$FONT_NAME"; then
        echo "✓ $FONT_NAME installed"
    else
        echo "✗ Failed to install $FONT_NAME"
    fi
}

# Install Noto fonts (best Unicode coverage)
if [ "$PKG_MANAGER" = "apt" ]; then
    if ! check_font "Noto Sans"; then
        echo "Installing Noto fonts..."
        sudo apt-get install -y fonts-noto fonts-noto-cjk fonts-noto-color-emoji
        echo "✓ Noto fonts installed"
    else
        echo "✓ Noto fonts already available"
    fi
elif [ "$PKG_MANAGER" = "brew" ]; then
    if ! check_font "Noto Sans"; then
        echo "Installing Noto fonts..."
        brew tap homebrew/cask-fonts
        brew install --cask font-noto-sans font-noto-mono font-noto-color-emoji
        echo "✓ Noto fonts installed"
    else
        echo "✓ Noto fonts already available"
    fi
fi

# ── Verify installation ───────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."

# Check all required tools
TOOLS=("pandoc" "xelatex" "pandoc-crossref")
ALL_GOOD=true

for tool in "${TOOLS[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        VERSION=$($tool --version | head -n1)
        echo "✓ $tool: $VERSION"
    else
        echo "✗ $tool: NOT FOUND"
        ALL_GOOD=false
    fi
done

# Check font availability
echo ""
echo "Font availability:"
if check_font "Arial"; then
    echo "✓ Arial"
elif check_font "Noto Sans"; then
    echo "✓ Noto Sans"
elif check_font "Liberation Sans"; then
    echo "✓ Liberation Sans"
elif check_font "DejaVu Sans"; then
    echo "✓ DejaVu Sans"
else
    echo "✗ No suitable main font found"
    ALL_GOOD=false
fi

if check_font "Consolas"; then
    echo "✓ Consolas"
elif check_font "Noto Sans Mono"; then
    echo "✓ Noto Sans Mono"
elif check_font "Liberation Mono"; then
    echo "✓ Liberation Mono"
elif check_font "DejaVu Sans Mono"; then
    echo "✓ DejaVu Sans Mono"
else
    echo "✗ No suitable mono font found"
    ALL_GOOD=false
fi

# ── Final status ─────────────────────────────────────────────────────────────────
echo ""
if [ "$ALL_GOOD" = true ]; then
    echo "🎉 All dependencies installed successfully!"
    echo ""
    echo "You can now use the md2pdf.sh script:"
    echo "  ./code/md2pdf.sh your-file.md"
    echo ""
    echo "For additional options:"
    echo "  ./code/md2pdf.sh your-file.md --docx --tex --verbose"
else
    echo "⚠️  Some dependencies may be missing."
    echo "Please check the output above and install any missing packages manually."
    exit 1
fi
