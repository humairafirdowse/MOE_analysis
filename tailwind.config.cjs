/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#050816",
        sidebar: "#050816",
        surface: "#0f172a",
        card: "#020617",
        accent: "#38bdf8",
        accentSoft: "#0ea5e9",
        accentMuted: "#1d4ed8",
        borderSoft: "#1e293b",
        textMuted: "#9ca3af"
      },
      boxShadow: {
        soft: "0 18px 45px rgba(15,23,42,0.85)"
      }
    }
  },
  plugins: []
};

