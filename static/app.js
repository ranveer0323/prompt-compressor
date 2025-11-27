// static/app.js
(async function(){
  // DOM refs
  const promptInput = document.getElementById('promptInput');
  const completeFlowBtn = document.getElementById('completeFlowBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const pruneBtn = document.getElementById('pruneBtn');
  const hybridBtn = document.getElementById('hybridBtn');
  const generateAllBtn = document.getElementById('generateAllBtn');
  // const wordsTableBody = document.querySelector('#wordsTable tbody');
  const phraseTableBody = document.querySelector('#phraseTable tbody');
  const iterLogDiv = document.getElementById('iterLog');
  const origPromptEl = document.getElementById('origPrompt');
  const prunedPromptEl = document.getElementById('prunedPrompt');
  const hybridPromptEl = document.getElementById('hybridPrompt');
  const origOutputEl = document.getElementById('origOutput');
  const prunedOutputEl = document.getElementById('prunedOutput');
  const hybridOutputEl = document.getElementById('hybridOutput');
  const origTokensEl = document.getElementById('origTokens');
  const prunedTokensEl = document.getElementById('prunedTokens');
  const hybridTokensEl = document.getElementById('hybridTokens');
  const pruneSimEl = document.getElementById('pruneSim');
  const hybridSimEl = document.getElementById('hybridSim');
  const keepRatio = document.getElementById('keepRatio');
  const maxPhraseLen = document.getElementById('maxPhraseLen');
  const simThreshold = document.getElementById('simThreshold');
  const downloadJson = document.getElementById('downloadJson');

  let lastRun = null;
  let surprisalChart = null;

  function jsonFetch(url, body){
    return fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
      .then(r => r.json());
  }

  // Update keep ratio display
  keepRatio.addEventListener('input', () => {
    const percent = Math.round(parseFloat(keepRatio.value) * 100);
    document.getElementById('keepRatioValue').textContent = percent + '%';
  });

  // Complete flow: analyze → prune → hybrid → generate
  completeFlowBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if(!prompt){ alert('Paste a prompt first'); return; }
    
    completeFlowBtn.disabled = true;
    completeFlowBtn.innerHTML = '<span>⏳ Running...</span>';
    
    try {
      // Step 1: Analyze
      console.log('Step 1: Analyzing...');
      clearUI();
      const analyzeRes = await jsonFetch('/api/analyze', { prompt });
      if(analyzeRes.error){ throw new Error(analyzeRes.error); }
      renderAnalysis(analyzeRes);
      origPromptEl.textContent = prompt;
      origTokensEl.textContent = analyzeRes.token_count || '-';
      
      // Step 2: Phrase Prune
      console.log('Step 2: Phrase pruning...');
      const pruneBody = { 
        prompt, 
        keep_ratio: parseFloat(keepRatio.value), 
        max_phrase_len: parseInt(maxPhraseLen.value), 
        sim_threshold: parseFloat(simThreshold.value) 
      };
      const pruneRes = await jsonFetch('/api/prune', pruneBody);
      if(pruneRes.error){ throw new Error(pruneRes.error); }
      lastRun = pruneRes.run_id;
      prunedPromptEl.textContent = pruneRes.compressed || '';
      prunedTokensEl.textContent = pruneRes.compressed ? pruneRes.compressed.split(/\s+/).length : '-';
      pruneSimEl.textContent = pruneRes.sim ? pruneRes.sim.toFixed(4) : '-';
      renderIterLog(pruneRes.log || []);
      
      // Step 3: Hybrid Compress
      console.log('Step 3: Hybrid compressing...');
      const hybridBody = { 
        prompt, 
        sim_threshold: parseFloat(simThreshold.value), 
        keep_ratio_phrases: parseFloat(keepRatio.value) 
      };
      const hybridRes = await jsonFetch('/api/hybrid', hybridBody);
      if(hybridRes.error){ throw new Error(hybridRes.error); }
      hybridPromptEl.textContent = hybridRes.hybrid || '';
      hybridTokensEl.textContent = hybridRes.hybrid ? hybridRes.hybrid.split(/\s+/).length : '-';
      
      // Step 4: Generate with Gemini
      console.log('Step 4: Generating with Gemini...');
      origOutputEl.textContent = 'Generating...';
      prunedOutputEl.textContent = 'Generating...';
      hybridOutputEl.textContent = 'Generating...';
      
      const originalResp = await jsonFetch('/api/generate', { which:'original', prompt });
      origOutputEl.textContent = originalResp.text || JSON.stringify(originalResp, null, 2);
      
      const prunedResp = await jsonFetch('/api/generate', { which:'pruned', run_id: lastRun });
      prunedOutputEl.textContent = prunedResp.text || JSON.stringify(prunedResp, null, 2);
      
      const hybridResp = await jsonFetch('/api/generate', { which:'hybrid', run_id: lastRun });
      hybridOutputEl.textContent = hybridResp.text || JSON.stringify(hybridResp, null, 2);
      
      // Step 5: Validate similarity
      console.log('Step 5: Computing similarities...');
      try {
        const simP = await jsonFetch('/api/validate', { a: originalResp.text || '', b: prunedResp.text || '' });
        const simH = await jsonFetch('/api/validate', { a: originalResp.text || '', b: hybridResp.text || '' });
        pruneSimEl.textContent = simP.similarity ? simP.similarity.toFixed(4) : '-';
        hybridSimEl.textContent = simH.similarity ? simH.similarity.toFixed(4) : '-';
      } catch(e){
        console.warn('validate failed', e);
      }
      
      console.log('✅ Complete flow finished!');
      alert('✅ Complete flow finished! Check the tabs for results.');
      
    } catch(e) {
      console.error('Complete flow error:', e);
      alert('❌ Error during complete flow: ' + e.message);
    } finally {
      completeFlowBtn.disabled = false;
      completeFlowBtn.innerHTML = '<span>🚀 Complete Flow</span>';
    }
  });

  analyzeBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if(!prompt){ alert('Paste a prompt first'); return; }
    clearUI();
    try {
      const res = await jsonFetch('/api/analyze', { prompt });
      console.log('Analyze response:', res);
      if(res.error){ alert('Error: ' + res.error); return; }
      renderAnalysis(res);
      origPromptEl.textContent = prompt;
      origTokensEl.textContent = res.token_count || '-';
    } catch(e) {
      console.error('Analyze failed:', e);
      alert('Error during analysis: ' + e.message);
    }
  });

  pruneBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if(!prompt){ alert('Paste a prompt first'); return; }
    clearUI();
    const body = { prompt, keep_ratio: parseFloat(keepRatio.value), max_phrase_len: parseInt(maxPhraseLen.value), sim_threshold: parseFloat(simThreshold.value) };
    const res = await jsonFetch('/api/prune', body);
    if(res.error){ alert(res.error); return; }
    // update UI
    lastRun = res.run_id;
    prunedPromptEl.textContent = res.compressed || '';
    prunedOutputEl.textContent = '(press Generate All to call Gemini)';
    prunedTokensEl.textContent = res.compressed ? res.compressed.split(/\s+/).length : '-';
    pruneSimEl.textContent = (res.sim || '-').toFixed ? (res.sim||'-') : res.sim;
    renderIterLog(res.log || []);
    // store JSON for download
    downloadJson.href = URL.createObjectURL(new Blob([JSON.stringify(res, null, 2)], {type:'application/json'}));
    downloadJson.download = 'prune_run_'+Date.now()+'.json';
  });

  hybridBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if(!prompt){ alert('Paste a prompt first'); return; }
    clearUI();
    const body = { prompt, sim_threshold: parseFloat(simThreshold.value), keep_ratio_phrases: parseFloat(keepRatio.value) };
    const res = await jsonFetch('/api/hybrid', body);
    if(res.error){ alert(res.error); return; }
    lastRun = res.run_id;
    hybridPromptEl.textContent = res.hybrid || '';
    hybridTokensEl.textContent = res.hybrid ? res.hybrid.split(/\s+/).length : '-';
    hybridSimEl.textContent = '-';
    downloadJson.href = URL.createObjectURL(new Blob([JSON.stringify(res, null, 2)], {type:'application/json'}));
    downloadJson.download = 'hybrid_run_'+Date.now()+'.json';
  });

  generateAllBtn.addEventListener('click', async () => {
    if(!promptInput.value.trim()){ alert('Paste a prompt first'); return; }
    // generate for original
    origOutputEl.textContent = 'Generating...';
    prunedOutputEl.textContent = 'Generating...';
    hybridOutputEl.textContent = 'Generating...';
    // build run payloads — prefer lastRun runs if present
    // original generation
    const originalResp = await jsonFetch('/api/generate', { which:'original', prompt: promptInput.value });
    origOutputEl.textContent = originalResp.text || JSON.stringify(originalResp, null, 2);
    // pruned: use lastRun if exists
    if(lastRun){
      const prunedResp = await jsonFetch('/api/generate', { which:'pruned', run_id: lastRun });
      prunedOutputEl.textContent = prunedResp.text || JSON.stringify(prunedResp, null, 2);
      const hybridResp = await jsonFetch('/api/generate', { which:'hybrid', run_id: lastRun });
      hybridOutputEl.textContent = hybridResp.text || JSON.stringify(hybridResp, null, 2);
      // compute similarity via validate endpoint
      try {
        const simP = await jsonFetch('/api/validate', { a: originalResp.text || '', b: prunedResp.text || '' });
        const simH = await jsonFetch('/api/validate', { a: originalResp.text || '', b: hybridResp.text || '' });
        pruneSimEl.textContent = simP.similarity ? simP.similarity.toFixed(4) : '-';
        hybridSimEl.textContent = simH.similarity ? simH.similarity.toFixed(4) : '-';
      } catch(e){
        console.warn('validate failed', e);
      }
    } else {
      // generate pruned/hybrid directly from prompt strings in UI
      const prunedText = prunedPromptEl.textContent || '';
      if(prunedText) {
        const prunedResp = await jsonFetch('/api/generate', { which:'pruned', prompt: prunedText });
        prunedOutputEl.textContent = prunedResp.text || JSON.stringify(prunedResp, null, 2);
      }
      const hybridText = hybridPromptEl.textContent || '';
      if(hybridText) {
        const hybridResp = await jsonFetch('/api/generate', { which:'hybrid', prompt: hybridText });
        hybridOutputEl.textContent = hybridResp.text || JSON.stringify(hybridResp, null, 2);
      }
    }
  });

  // helpers
  function renderAnalysis(res){
    try {
      // words not full details; we expect top_phrase_candidates
      // wordsTableBody.innerHTML = '';
      phraseTableBody.innerHTML = '';

      const candidates = res.top_phrase_candidates || [];
      console.log('Candidates count:', candidates.length);
      
      if(candidates.length === 0) {
        phraseTableBody.innerHTML = '<tr><td colspan="2" class="text-muted">No candidates found</td></tr>';
        return;
      }

      // render phrase candidates (first 200)
      for(const p of candidates.slice(0,200)){
        const row = document.createElement('tr');
        row.innerHTML = `<td>${escapeHtml(p.phrase || '')}</td><td class="text-end">${(p.norm||0).toFixed(4)}</td>`;
        phraseTableBody.appendChild(row);
      }

      // draw a simple chart based on norms of candidates
      const norms = candidates.slice(0,60).map(x=>x.norm||0);
      if(norms.length > 0) {
        drawChart(norms);
      }
    } catch(e) {
      console.error('renderAnalysis error:', e);
      phraseTableBody.innerHTML = '<tr><td colspan="2" class="text-danger">Error rendering analysis</td></tr>';
    }
  }

  function renderIterLog(log){
    iterLogDiv.innerHTML = '';
    if(!log || log.length === 0){ iterLogDiv.textContent = '(no iterations or nothing removable)'; return; }
    log.forEach(item => {
      const div = document.createElement('div');
      div.className = 'mb-1';
      div.innerHTML = `<strong>Iter ${item.iter}</strong> — removed: <em>${escapeHtml(item.removed_phrase)}</em> — words now: ${item.current_word_count}`;
      iterLogDiv.appendChild(div);
    });
  }

  function drawChart(values){
    try {
      const canvas = document.getElementById('surprisalChart');
      if(!canvas) {
        console.error('Chart canvas not found');
        return;
      }
      const ctx = canvas.getContext('2d');
      if(surprisalChart) surprisalChart.destroy();
      surprisalChart = new Chart(ctx, {
        type: 'bar',
        data: { 
          labels: values.map((_,i)=>i+1), 
          datasets: [{ 
            label:'norm surprisal', 
            data: values,
            backgroundColor: 'rgba(13, 110, 253, 0.6)',
            borderColor: 'rgba(13, 110, 253, 1)',
            borderWidth: 1
          }] 
        },
        options: { 
          responsive:true, 
          maintainAspectRatio: true,
          plugins:{ legend:{display:false} }, 
          scales:{ x:{ display:false }, y:{ beginAtZero:true } } 
        }
      });
    } catch(e) {
      console.error('drawChart error:', e);
    }
  }

  function clearUI(){
    phraseTableBody.innerHTML = '';
    iterLogDiv.innerHTML = '';
    origOutputEl.textContent = '';
    prunedOutputEl.textContent = '';
    hybridOutputEl.textContent = '';
    prunedPromptEl.textContent = '';
    hybridPromptEl.textContent = '';
    prunedTokensEl.textContent = '-';
    hybridTokensEl.textContent = '-';
    pruneSimEl.textContent = '-';
    hybridSimEl.textContent = '-';
  }

  function escapeHtml(s){ if(!s) return ''; return s.replace(/[&<>"']/g, (c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

})();
