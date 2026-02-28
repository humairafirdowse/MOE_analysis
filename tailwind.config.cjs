/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#06080f",
        sidebar: "#080b14",
        surface: "#0d1425",
        card: "#0a0f1e",
        accent: "#38bdf8",
        accentSoft: "#0ea5e9",
        accentMuted: "#1d4ed8",
        borderSoft: "#1a2540",
        textMuted: "#8494b0",
        sectionArch: "#6366f1",
        sectionMoe: "#a855f7",
        sectionTrain: "#f59e0b",
        sectionInfer: "#10b981",
        sectionPreset: "#38bdf8",
        kpiBlue: "#38bdf8",
        kpiGreen: "#34d399",
        kpiAmber: "#fbbf24",
        kpiPurple: "#a78bfa",
        kpiRose: "#fb7185"
      },
      boxShadow: {
        soft: "0 18px 45px rgba(6,8,15,0.9)",
        glow: "0 0 20px rgba(56,189,248,0.08)",
        cardHover: "0 4px 20px rgba(56,189,248,0.06)"
      }
    }
  },
  plugins: []
};
