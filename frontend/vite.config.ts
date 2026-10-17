import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dashboard talks to the FastAPI backend (python -m backend.api.main) through this proxy in development.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 1600 },
});
