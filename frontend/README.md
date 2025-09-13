# Risk Simulation Frontend

A modern React-based frontend for the Monte Carlo Risk Simulation Engine.

## Features

🎨 **Modern UI/UX**
- Beautiful gradient backgrounds and glassmorphism effects
- Responsive design that works on all devices
- Smooth animations with Framer Motion
- Interactive charts with Recharts

🔧 **Simulation Controls**
- Choose between Python and Node.js engines
- Support for 5 simulation methods:
  - Monte Carlo
  - Historical Simulation
  - Bootstrap
  - Variance-Covariance
  - Geometric Brownian Motion

📊 **Rich Results Display**
- Risk metrics (VaR, CVaR, Expected Return, Volatility)
- Interactive distribution charts
- Performance metrics
- Error handling with user-friendly messages

## Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **Recharts** - Interactive charts
- **Lucide React** - Beautiful icons

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## Configuration

The frontend expects the backend API to be running on `http://localhost:5010`. 

Update the API endpoint in `src/App.jsx` if your backend runs on a different port:

```javascript
const res = await fetch(`http://localhost:YOUR_PORT/simulate`, {
```

## Usage

1. **Select Engine**: Choose between Python or Node.js backend
2. **Choose Method**: Pick from 5 different simulation methods
3. **Configure Portfolio**: Enter comma-separated portfolio values
4. **Set Parameters**: Adjust iterations and confidence level
5. **Run Simulation**: Click the button and view results

## API Integration

The frontend sends POST requests to `/simulate` with the following structure:

```json
{
  "engine": "python|node",
  "method": "monteCarlo|historical|bootstrap|varcov|gbm",
  "portfolio": [100000, 200000, 150000],
  "params": {
    "iterations": 100000,
    "confidence": 0.95
  }
}
```

## Development

- **Hot Reload**: Changes are instantly reflected during development
- **TypeScript Ready**: Easy to migrate to TypeScript if needed
- **Component Library**: Reusable UI components in `src/components/ui/`
- **Responsive**: Works on desktop, tablet, and mobile devices

## Customization

- **Styling**: Modify `tailwind.config.js` for custom themes
- **Components**: Add new UI components in `src/components/ui/`
- **Charts**: Customize chart appearance in the `ResponsiveContainer` section
- **Animations**: Adjust Framer Motion animations for different effects
