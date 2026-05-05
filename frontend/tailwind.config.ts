import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sidebar: "#0f172a",
        sidebarText: "#cbd5e1",
        sidebarActive: "#1e293b",
        accent: {
          DEFAULT: "#4f46e5",
          hover: "#4338ca",
        },
        surface: "#ffffff",
        bg: "#f8fafc",
        border: "#e2e8f0",
        muted: "#64748b",
      },
      borderRadius: {
        xl: "0.875rem",
      },
    },
  },
  plugins: [],
};

export default config;
