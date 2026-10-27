/** Lightweight browser dashboard demo; backend integration can be added via a documented API. */
type Metric = { label: string; value: string; detail: string };
const metrics: Metric[] = [
  { label: "Engineering calculations", value: "11+", detail: "Existing Python toolkit" },
  { label: "Analytics", value: "EDA", detail: "CSV summaries and correlations" },
  { label: "Monitoring", value: "Anomaly flags", detail: "Robust median/MAD baseline" },
  { label: "Simulation", value: "ODE", detail: "Mass-spring vibration model" },
];
const root = document.querySelector<HTMLElement>("#app");
if (root) {
  root.innerHTML = `<header><h1>Mechanical Engineering Data Lab</h1><p>Engineering analysis • data science • simulation</p></header><main class="grid">${metrics.map(m => `<article><h2>${m.label}</h2><strong>${m.value}</strong><p>${m.detail}</p></article>`).join("")}</main><footer>Demo frontend — calculations run in the Python desktop/web tools.</footer>`;
}
