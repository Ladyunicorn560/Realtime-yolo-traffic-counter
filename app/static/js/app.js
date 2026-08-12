document.addEventListener('DOMContentLoaded', () => {
    const videoFeed  = document.getElementById('video-feed');
    const loading    = document.getElementById('loading');
    const status     = document.getElementById('status');
    const totalCount = document.getElementById('total-count');
    const upCount    = document.getElementById('up-count');
    const downCount  = document.getElementById('down-count');

    let statsInterval = null;
    let frameLooping  = false;   // true while the chained frame loop is active

    // ── Chained frame loader ──────────────────────────────────────────────────
    // Each request fires AFTER the previous one completes (load or error).
    // This prevents request pile-up when the server is busy with YOLO inference.
    const startFrameLoop = () => {
        if (frameLooping) return;
        frameLooping = true;

        const loadNext = () => {
            if (!frameLooping) return;
            const img = new Image();
            img.onload = () => {
                videoFeed.src = img.src;          // swap only when ready — no flicker
                requestAnimationFrame(loadNext);   // schedule next right after paint
            };
            img.onerror = () => {
                if (frameLooping) setTimeout(loadNext, 100);  // retry after brief pause
            };
            img.src = '/frame?t=' + Date.now();
        };

        loadNext();
    };

    const stopFrameLoop = () => { frameLooping = false; };

    const setRunning = (isRunning, msg = 'Camera Live') => {
        if (isRunning) {
            loading.style.display   = 'none';
            videoFeed.style.display = 'block';
            status.innerText        = msg;
            startFrameLoop();
            if (!statsInterval) statsInterval = setInterval(updateStats, 500);
        } else {
            stopFrameLoop();
            if (statsInterval) { clearInterval(statsInterval); statsInterval = null; }
            videoFeed.removeAttribute('src');
            videoFeed.style.display = 'none';
            loading.style.display   = 'flex';
            loading.innerText       = 'Select a source to start...';
            status.innerText        = 'Stopped';
        }
    };

    const updateStats = async () => {
        try {
            const res = await fetch('/stats');
            if (!res.ok) return;
            const d = await res.json();
            totalCount.innerText = d.total;
            upCount.innerText    = d.up;
            downCount.innerText  = d.down;
            document.getElementById('car-count').innerText       = d.car;
            document.getElementById('motorbike-count').innerText = d.motorbike;
            document.getElementById('bus-count').innerText       = d.bus;
            document.getElementById('truck-count').innerText     = d.truck;
        } catch (e) {}
    };

    const DEFAULT_SOURCE = "rtsp://admin:Kiran%4011@192.168.1.64:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1";

    const startCamera = async (source) => {
        source = source || document.getElementById('cam-source').value || DEFAULT_SOURCE;
        status.innerText = 'Connecting...';
        try {
            const res = await fetch('/start', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ source })
            });
            if (res.ok) setRunning(true, 'Camera Live');
            else        status.innerText = 'Server Error';
        } catch (e) {
            status.innerText = 'Connection Error';
        }
    };

    document.getElementById('start-cam').onclick = () =>
        startCamera(document.getElementById('cam-source').value);

    document.getElementById('stop-cam').onclick = async () => {
        await fetch('/stop', { method: 'POST' });
        setRunning(false);
    };

    document.getElementById('video-upload').onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        status.innerText  = 'Uploading...';
        loading.innerText = 'Uploading Video...';
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res = await fetch('/upload_video', { method: 'POST', body: formData });
            if (res.ok) setRunning(true, 'Analyzing Video');
            else { status.innerText = 'Upload Error'; loading.innerText = 'Upload failed.'; }
        } catch (err) {
            status.innerText  = 'Connection Error';
            loading.innerText = 'Error connecting to server.';
        }
    };

    // ── Webhook ───────────────────────────────────────────────────────────────

    const webhookStatus = document.getElementById('webhook-status');

    // Load saved URL on page start
    (async () => {
        try {
            const res = await fetch('/webhook/config');
            if (!res.ok) return;
            const cfg = await res.json();
            if (cfg.url) document.getElementById('webhook-url').value = cfg.url;
        } catch (e) {}
    })();

    document.getElementById('webhook-save').onclick = async () => {
        const url = document.getElementById('webhook-url').value.trim();
        try {
            const res = await fetch('/webhook/config', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ url })
            });
            const data = await res.json();
            webhookStatus.textContent = data.status === 'enabled' ? '✅ Saved' : '⚠️ Cleared';
            webhookStatus.style.color = data.status === 'enabled' ? '#34d399' : '#9ca3af';
            setTimeout(() => { webhookStatus.textContent = ''; }, 2500);
        } catch (e) {
            webhookStatus.textContent = '❌ Error';
            webhookStatus.style.color = '#f87171';
        }
    };

    document.getElementById('webhook-test').onclick = async () => {
        const url = document.getElementById('webhook-url').value.trim();
        // Auto-save first so the backend always has the latest URL
        if (url) {
            await fetch('/webhook/config', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ url })
            });
        }
        webhookStatus.textContent = '🔄 Sending test…';
        webhookStatus.style.color = '#93c5fd';
        try {
            const res  = await fetch('/webhook/test', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                webhookStatus.textContent = `✅ Test sent! HTTP ${data.http_code}`;
                webhookStatus.style.color = '#34d399';
            } else {
                webhookStatus.textContent = `❌ ${data.message}`;
                webhookStatus.style.color = '#f87171';
            }
        } catch (e) {
            webhookStatus.textContent = '❌ Network error';
            webhookStatus.style.color = '#f87171';
        }
        setTimeout(() => { webhookStatus.textContent = ''; }, 4000);
    };

    // Auto-connect on page load
    startCamera(DEFAULT_SOURCE);
});

