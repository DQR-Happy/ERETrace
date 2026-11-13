(() => {
  const root = document.documentElement;
  let saved;
  try { saved = localStorage.getItem('eretrace-theme'); } catch (_) {}
  root.dataset.theme = saved || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.querySelector('.theme').addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('eretrace-theme', next); } catch (_) {}
  });
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);
        button.textContent = 'Copied';
      } catch (_) { button.textContent = 'Select to copy'; }
      setTimeout(() => { button.textContent = 'Copy'; }, 1800);
    });
  });
  const model = document.getElementById('model');
  fetch('assets/results.json').then(response => {
    if (!response.ok) throw new Error('Results unavailable');
    return response.json();
  }).then(data => {
    function update() {
      const body = document.getElementById('result-rows');
      body.replaceChildren();
      for (const row of data[model.value]) {
        const tr = document.createElement('tr');
        if (row.id === 'R') tr.className = 'ours';
        const th = document.createElement('th'); th.scope = 'row'; th.textContent = row.method;
        const ms = document.createElement('td'); ms.textContent = row.ms.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        const gb = document.createElement('td'); gb.textContent = row.id === 'N' ? '—' : row.gb.toFixed(3);
        tr.append(th, ms, gb); body.append(tr);
      }
      document.getElementById('result-caption').textContent = model.value + ' · same model, workload and capture scope.';
    }
    model.addEventListener('change', update); update();
  }).catch(() => { model.disabled = true; model.title = 'Showing the embedded Qwen3-8B results'; });
})();
