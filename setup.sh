#!/bin/bash
# DDQ Platform - Quick setup script

echo "Setting up DDQ Platform..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"
echo ""
echo "Next steps:"
echo "  1. Add your Anthropic API key to .env:"
echo "     ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  2. Set your fund name in .env:"
echo "     FUND_NAME=Your Fund Name"
echo ""
echo "  3. Run the app:"
echo "     source venv/bin/activate"
echo "     python app.py"
echo ""
echo "  4. Open http://localhost:5000"
echo "     Login: admin@fund.com / admin123"
echo ""
echo "  5. Upload fund documents (LPA, presentations, policies)"
echo "     Or place files in /documents and use the bulk ingest button"
