# VS Code Debugger Configuration Guide

## 🚀 Updated Debugger Configurations

### 🎯 **QUICK START - Recommended Options:**

1. **"Launch Full Stack (Python + Node.js + Frontend)"** - Starts everything at once
2. **"Node.js Backend Server (Fixed)"** - Uses the stable `index-fixed.js` file
3. **"Launch Both Backends (Fixed)"** - Starts both backends with fixed Node.js server

---

## 🔧 Individual Server Configurations

### 🐍 **Python Backend**
- **"Python Backend Server"** - Main Flask server
- **"Debug Current Python File"** - Debug any Python file

### 🟢 **Node.js Backend**
- **"Node.js Backend Server (Fixed)"** ✅ - Uses `index-fixed.js` with enhanced error handling
- **"Node.js Backend Server (Original)"** - Uses original `index.js` file

### ⚛️ **React Frontend**
- **"React Frontend Server"** - Direct Vite server launch
- **"React Frontend (NPM)"** - Uses npm script (recommended)

---

## 🚀 Compound Configurations (Multiple Services)

### 🎯 **Full Stack Development**
- **"Launch Full Stack (Python + Node.js + Frontend)"** - All three services
  - Python Backend: port 3002
  - Node.js Backend: port 5010
  - React Frontend: port 3000

### 🔄 **Backend Development**
- **"Launch Both Backends (Fixed)"** - Both backends with stable Node.js
- **"Launch Both Backends (Original)"** - Both backends with original Node.js

### 🎨 **Frontend Development**
- **"Launch Frontend + Python Backend"** - Frontend with Python API
- **"Launch Frontend + Node.js Backend"** - Frontend with Node.js API

---

## 🔍 **How to Use:**

1. **Open VS Code**
2. **Press `F5`** or go to **Run and Debug** (Ctrl+Shift+D)
3. **Select configuration** from dropdown
4. **Click Play button** or press F5

### 🎯 **For Your Use Case:**
Since you want both backends and frontend from debugger, use:
**"Launch Full Stack (Python + Node.js + Frontend)"**

This will start:
- ✅ Python Flask server on port 3002
- ✅ Node.js Express server on port 5010 (using fixed version)
- ✅ React frontend on port 3000

---

## 🛠️ **Key Fixes Applied:**

1. **Node.js Debugger Fix**: Changed from `index.js` to `index-fixed.js`
2. **Added --expose-gc**: Enables garbage collection for better memory management
3. **Frontend Integration**: Added React debugging configurations
4. **Compound Configs**: Easy one-click launch for multiple services
5. **Task Integration**: Added frontend dependency installation

---

## 🔧 **Files Modified:**

- `.vscode/launch.json` - Debugger configurations
- `.vscode/tasks.json` - Build and install tasks
- `backend/node/index-fixed.js` - Stable Node.js server
- `backend/node/package.json` - Updated to use fixed server

---

## 🚨 **Important Notes:**

- Always use **"Node.js Backend Server (Fixed)"** for debugging Node.js
- The fixed version has enhanced error handling and won't crash
- Frontend configurations will install dependencies automatically
- All services run in separate integrated terminals for easy monitoring

---

## 🎉 **You're All Set!**

Now you can debug both backends and frontend directly from VS Code with proper error handling and stability!
