/* Home page: pinned training pipeline, loss curve, draggable photo strip.
   Everything here is progressive enhancement: without it the page is a plain, readable column. */
(function () {
  var line = document.getElementById('line');
  var hero = document.getElementById('hero');
  if (!line || !hero) return;

  var stages = [].slice.call(line.querySelectorAll('.stage'));
  var ticks = [].slice.call(line.querySelectorAll('.tick'));
  var N = stages.length;
  if (!N) return;

  /* The loss curve: exponential decay with one small spike after the second station
     (the pretraining article is the one about loss spikes). */
  var spikeAt = N >= 2 ? 1.72 / N : 0.2;
  function loss(x) { return Math.exp(-3.4 * x) + 0.16 * Math.exp(-Math.pow((x - spikeAt) / 0.012, 2)); }
  function curveY(x) { return 0.92 - 0.84 * Math.min(loss(x), 1); }      /* 0..1; the top is high loss */
  var d = '';
  for (var k = 0; k <= 400; k++) {
    var x = k / 400;
    d += (k ? 'L' : 'M') + (x * 1000).toFixed(1) + ' ' + (curveY(x) * 100).toFixed(2);
  }
  var track = document.getElementById('curve-track');
  var drawn = document.getElementById('curve-drawn');
  if (track) track.setAttribute('d', d);
  if (drawn) drawn.setAttribute('d', d);

  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var roomy = window.matchMedia('(min-width: 860px) and (min-height: 560px)');
  var active = -1;

  function setMode() {
    var pinned = roomy.matches && !reduce.matches;
    line.classList.toggle('is-pinned', pinned);
    ticks.forEach(function (t) { t.tabIndex = pinned ? 0 : -1; });
    if (!pinned) {
      active = -1;
      stages.forEach(function (s) { s.classList.remove('is-active'); });
    }
    update();
  }

  function update() {
    var vh = window.innerHeight;
    var hr = hero.getBoundingClientRect();
    hero.style.setProperty('--h', Math.max(0, Math.min(1, -hr.top / (hr.height * 0.8))).toFixed(3));
    if (!line.classList.contains('is-pinned')) return;

    var r = line.getBoundingClientRect();
    var p = Math.max(0, Math.min(1, -r.top / (r.height - vh)));
    line.style.setProperty('--p', p.toFixed(4));
    line.style.setProperty('--dx', p.toFixed(4));
    line.style.setProperty('--dy', curveY(p).toFixed(4));

    var i = Math.min(N - 1, Math.floor(p * N));
    if (i !== active) {
      active = i;
      line.style.setProperty('--hue', stages[i].style.getPropertyValue('--hue'));
      stages.forEach(function (s, j) { s.classList.toggle('is-active', j === i); });
      ticks.forEach(function (t, j) {
        t.classList.toggle('is-active', j === i);
        t.classList.toggle('is-done', j < i);
      });
    }
  }

  var queued = false;
  function onScroll() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; update(); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  if (roomy.addEventListener) {
    roomy.addEventListener('change', setMode);
    reduce.addEventListener('change', setMode);
  }

  ticks.forEach(function (t, j) {
    t.addEventListener('click', function () {
      var top = line.getBoundingClientRect().top + window.pageYOffset;
      window.scrollTo({ top: top + (j + 0.5) / N * (line.offsetHeight - window.innerHeight) });
    });
  });
  setMode();

  /* Photo strip: native horizontal scrolling already works for touch and trackpads;
     this adds click-and-drag for mouse users. */
  var strip = document.getElementById('strip');
  if (!strip) return;
  var down = false, moved = false, startX = 0, startLeft = 0;
  strip.addEventListener('pointerdown', function (e) {
    if (e.pointerType !== 'mouse' || e.button !== 0) return;
    down = true; moved = false; startX = e.clientX; startLeft = strip.scrollLeft;
  });
  window.addEventListener('pointermove', function (e) {
    if (!down) return;
    var dx = e.clientX - startX;
    if (!moved && Math.abs(dx) > 5) { moved = true; strip.classList.add('is-dragging'); }
    if (moved) strip.scrollLeft = startLeft - dx;
  });
  window.addEventListener('pointerup', function () { down = false; strip.classList.remove('is-dragging'); });
  /* a drag must not count as a click on the photo's link */
  strip.addEventListener('click', function (e) { if (moved) { e.preventDefault(); moved = false; } }, true);
})();
