// Simple hover tooltip for resume annotations
(function () {
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip-floating';
  tooltip.style.position = 'fixed';
  tooltip.style.pointerEvents = 'none';
  tooltip.style.display = 'none';
  document.body.appendChild(tooltip);

  function showTip(ev, el) {
    const tone = el.getAttribute('data-tone') || '';
    const req = el.getAttribute('data-req') || '';
    const comment = el.getAttribute('data-comment') || '';
    let html = '';
    if (req) html += `<div class="tip-req">${req}</div>`;
    if (comment) html += `<div class="tip-comment">${comment}</div>`;
    if (!html) html = '<div class="tip-comment">Без комментария</div>';
    tooltip.innerHTML = html;
    tooltip.setAttribute('data-tone', tone);
    tooltip.style.display = 'block';
    position(ev);
  }

  function hideTip() {
    tooltip.style.display = 'none';
  }

  function position(ev) {
    const pad = 12;
    const x = Math.min(window.innerWidth - tooltip.offsetWidth - pad, ev.clientX + pad);
    const y = Math.min(window.innerHeight - tooltip.offsetHeight - pad, ev.clientY + pad);
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
  }

  document.addEventListener('mouseover', (ev) => {
    const el = ev.target.closest('.resume-annotated .hl');
    if (el) showTip(ev, el);
  });
  document.addEventListener('mousemove', (ev) => {
    const el = ev.target.closest('.resume-annotated .hl');
    if (el && tooltip.style.display === 'block') position(ev);
  });
  document.addEventListener('mouseout', (ev) => {
    const el = ev.target.closest('.resume-annotated .hl');
    if (el) hideTip();
  });
})();

