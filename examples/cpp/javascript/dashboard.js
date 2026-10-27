// Plain JavaScript fallback frontend; open index.html in a browser.
const root = document.querySelector('#app');
const cards = [['Calculators','11+','Existing Python toolkit'],['Data analysis','EDA','CSV summaries and correlations'],['Anomaly detection','MAD','Robust baseline'],['Simulation','ODE','Mass-spring vibration']];
if (root) root.innerHTML = `<h1>Mechanical Engineering Data Lab</h1><div class="grid">${cards.map(([a,b,c])=>`<section><h2>${a}</h2><strong>${b}</strong><p>${c}</p></section>`).join('')}</div>`;
