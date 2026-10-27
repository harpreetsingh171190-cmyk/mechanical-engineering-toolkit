// Compiled browser equivalent of src/dashboard.ts for a no-build preview.
const metrics = [
  {label:'Engineering calculations', value:'11+', detail:'Existing Python toolkit'},
  {label:'Analytics', value:'EDA', detail:'CSV summaries and correlations'},
  {label:'Monitoring', value:'Anomaly flags', detail:'Robust median/MAD baseline'},
  {label:'Simulation', value:'ODE', detail:'Mass-spring vibration model'}
];
const root = document.querySelector('#app');
if (root) root.innerHTML = `<header><h1>Mechanical Engineering Data Lab</h1><p>Engineering analysis • data science • simulation</p></header><main class="grid">${metrics.map(m => `<article><h2>${m.label}</h2><strong>${m.value}</strong><p>${m.detail}</p></article>`).join('')}</main><footer>Demo frontend — calculations run in the Python desktop/web tools.</footer>`;
