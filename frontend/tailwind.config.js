/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eefdfb", 100: "#d4f7f2", 200: "#aceee8", 300: "#74dfd8",
          400: "#39c6c2", 500: "#15a9a8", 600: "#0e8789", 700: "#106c6e",
          800: "#125659", 900: "#13484b",
        },
        ink: {
          50: "#f6f8fa", 100: "#eef1f5", 200: "#dfe5ec", 300: "#c6cfdb",
          400: "#9aa7b8", 500: "#6f7e93", 600: "#556277", 700: "#434f61",
          800: "#2f3847", 900: "#1c2330",
        },
        risk: {
          low: "#16a34a", lowbg: "#ecfdf3", lowring: "#bbf7d0",
          mod: "#d97706", modbg: "#fffbeb", modring: "#fde68a",
          high: "#dc2626", highbg: "#fef2f2", highring: "#fecaca",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,0.04), 0 4px 16px rgba(16,24,40,0.06)",
        pop: "0 8px 30px rgba(16,24,40,0.12)",
      },
      borderRadius: { xl2: "1.125rem" },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
