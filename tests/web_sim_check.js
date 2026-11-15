// Runs the report's simulator in headless Chromium on the wheel: WebAssembly (run_cpu) against WebGPU (SwiftShader),
// which must fill the same histograms word for word, and the page's own button. Usage: node web_sim_check.js <site url>
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'] });
  const p = await b.newPage(); const errors = []; p.on('pageerror', (e) => errors.push(e.message));
  await p.goto(process.argv[2] + '/index.html?sim=webgpu#next');
  const runs = await p.evaluate(async () => {
    const { k, device } = await window.lottoSim.load();
    const n = D.assimilation.real.next.lines.length, out = [];  // load() staged the wheel's lines and the model's weights
    for (const weighted of [0, 1]) {
      const seed = 12345 + weighted, paths = 2000;
      const pc = k.run_cpu(n, weighted, paths, seed, 30), cpu = new Float32Array(k.memory.buffer, pc, 21).slice();
      const hc = new Uint32Array(k.memory.buffer, k.hist_ptr(), k.hist_words()).slice();
      let gpu = null, differing = null;
      if (device) {
        await window.lottoSim.gpu(k.gpu_prepare(n, weighted, paths, seed));
        const pg = k.gpu_finish(30); gpu = Array.from(new Float32Array(k.memory.buffer, pg, 21));
        const hg = new Uint32Array(k.memory.buffer, k.hist_ptr(), k.hist_words());
        differing = 0; for (let i = 0; i < hc.length; i++) differing += hc[i] !== hg[i];
      }
      out.push({ weighted, cpu: Array.from(cpu), gpu, differing, paths: hc.slice(0, 7).reduce((a, x) => a + x, 0) });
    }
    return out;
  });
  await p.click('#sim-run');
  await p.waitForFunction(() => !$('#sim-wrap').hidden || /Couldn't/.test($('#sim-status').textContent), null, { timeout: 120000 });
  const table = await p.$$eval('#sim-table tbody tr', (rows) => rows.length);
  console.log(JSON.stringify({ runs, table, status: await p.textContent('#sim-status'), errors }));
  await b.close();
})();
