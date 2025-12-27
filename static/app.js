/**
 * LLM-NMT Translation System - Frontend JavaScript
 * Handles translation requests, UI updates, and session management
 */

// ============================================
// Configuration
// ============================================

const API_BASE = '';  // Same origin
let sessionId = generateSessionId();
let currentContext = null;

// ============================================
// Utility Functions
// ============================================

function generateSessionId() {
    return 'session_' + Math.random().toString(36).substring(2, 15);
}

function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}

// ============================================
// API Functions
// ============================================

async function translateQuery(query) {
    try {
        const response = await fetch(`${API_BASE}/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId,
                include_metrics: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Translation error:', error);
        throw error;
    }
}

async function addSessionEvent(category) {
    try {
        await fetch(`${API_BASE}/session/${sessionId}/event`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                event_type: 'view_product',
                category: category
            })
        });
    } catch (error) {
        console.error('Session event error:', error);
    }
}

async function getMetricsSummary() {
    try {
        const response = await fetch(`${API_BASE}/metrics/summary`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Metrics error:', error);
        return null;
    }
}

async function getDemoQueries() {
    try {
        const response = await fetch(`${API_BASE}/demo/queries`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Demo queries error:', error);
        return null;
    }
}

// ============================================
// UI Update Functions
// ============================================

function setLoading(isLoading) {
    const btn = $('#translate-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    if (isLoading) {
        btn.disabled = true;
        btnText.classList.add('hidden');
        btnLoading.classList.remove('hidden');
    } else {
        btn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoading.classList.add('hidden');
    }
}

function updateOutput(result) {
    const outputEl = $('#english-output');

    // Build output HTML with entity highlighting
    let html = result.translation;

    // Highlight preserved entities
    if (result.entities_preserved && result.entities_preserved.length > 0) {
        result.entities_preserved.forEach(entity => {
            const regex = new RegExp(`(${escapeRegex(entity)})`, 'gi');
            html = html.replace(regex, '<span class="entity">$1</span>');
        });
    }

    outputEl.innerHTML = html;
    outputEl.classList.add('success', 'fade-in');

    // Remove animation class after it completes
    setTimeout(() => {
        outputEl.classList.remove('fade-in');
    }, 300);
}

function updateTierBreakdown(tierUsed) {
    const tiers = ['cache', 'entity', 'ambiguity', 'nmt'];

    tiers.forEach(tier => {
        const el = $(`#tier-${tier}`);
        el.classList.remove('active', tier);

        if (tier === tierUsed) {
            el.classList.add('active', tier);
        }
    });
}

function updateEntities(entities) {
    const container = $('#entities-list');

    if (!entities || entities.length === 0) {
        container.innerHTML = '<span class="placeholder">No entities detected</span>';
        return;
    }

    container.innerHTML = entities.map(entity =>
        `<span class="entity-tag">${entity}</span>`
    ).join('');
}

function updateAmbiguity(ambiguityInfo) {
    const container = $('#ambiguity-info');

    if (!ambiguityInfo || !ambiguityInfo.resolved_words ||
        Object.keys(ambiguityInfo.resolved_words).length === 0) {
        container.innerHTML = '<span class="placeholder">No ambiguous words</span>';
        return;
    }

    const resolutions = Object.entries(ambiguityInfo.resolved_words);
    container.innerHTML = resolutions.map(([word, meaning]) => `
        <div class="resolution">
            <span class="word">${word}</span>
            <span class="arrow">→</span>
            <span class="meaning">${meaning}</span>
        </div>
    `).join('');
}

function updateMetrics(result) {
    $('#latency-value').textContent = `${result.latency_ms.toFixed(0)}ms`;
    $('#confidence-value').textContent = `${(result.confidence * 100).toFixed(0)}%`;
    $('#tier-value').textContent = result.tier_used.charAt(0).toUpperCase() + result.tier_used.slice(1);
}

async function updateSystemStats() {
    const stats = await getMetricsSummary();
    if (!stats) return;

    $('#stat-total').textContent = stats.total_translations;
    $('#stat-cache-rate').textContent = `${(stats.cache_hit_rate * 100).toFixed(0)}%`;
    $('#stat-avg-latency').textContent = `${stats.average_latency_ms.toFixed(0)}ms`;
    $('#stat-entities').textContent = stats.entity_preservation_count;
}

function updateContext(category) {
    currentContext = category;

    // Update button states
    $$('.context-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.category === category) {
            btn.classList.add('active');
        }
    });

    // Update display
    const categoryNames = {
        'school_supplies': '📚 School Supplies',
        'candy': '🍬 Candy',
        'fruit': '🍑 Fruits',
        'sports': '🎣 Sports',
        'cosmetics': '💄 Cosmetics',
        'dairy': '🥛 Dairy'
    };

    $('#context-value').textContent = categoryNames[category] || category;
}

async function loadDemoQueries() {
    const data = await getDemoQueries();
    if (!data || !data.queries) {
        // Fallback demo queries
        renderDemoQueries([
            { french: "yogurt liberté logo", expected: "yogurt liberté logo", feature: "Entity preservation" },
            { french: "je veux gomme", expected: "i want eraser", feature: "Ambiguity resolution" },
            { french: "papier royale", expected: "paper royale", feature: "Brand preservation" },
            { french: "pêche fraîche", expected: "fresh peach", feature: "Context translation" }
        ]);
        return;
    }

    renderDemoQueries(data.queries);
}

function renderDemoQueries(queries) {
    const container = $('#demo-queries');

    container.innerHTML = queries.map(q => `
        <div class="demo-query" data-query="${escapeHtml(q.french)}">
            <div class="french">${escapeHtml(q.french)}</div>
            <div class="expected">→ ${escapeHtml(q.expected)}</div>
            <span class="feature">${escapeHtml(q.feature)}</span>
        </div>
    `).join('');

    // Add click handlers
    $$('.demo-query').forEach(el => {
        el.addEventListener('click', () => {
            const query = el.dataset.query;
            $('#french-input').value = query;
            handleTranslate();
        });
    });
}

// ============================================
// Helper Functions
// ============================================

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Event Handlers
// ============================================

async function handleTranslate() {
    const query = $('#french-input').value.trim();

    if (!query) {
        return;
    }

    setLoading(true);

    try {
        const result = await translateQuery(query);

        updateOutput(result);
        updateTierBreakdown(result.tier_used);
        updateEntities(result.entities_preserved);
        updateAmbiguity(result.ambiguity_resolved);
        updateMetrics(result);

        // Update stats after translation
        updateSystemStats();

    } catch (error) {
        $('#english-output').innerHTML = `<span style="color: var(--error)">Translation failed: ${error.message}</span>`;
    } finally {
        setLoading(false);
    }
}

async function handleContextClick(category) {
    updateContext(category);

    // Send multiple events to build context
    for (let i = 0; i < 3; i++) {
        await addSessionEvent(category);
    }
}

// ============================================
// Initialization
// ============================================

function init() {
    // Translate button
    $('#translate-btn').addEventListener('click', handleTranslate);

    // Enter key in textarea
    $('#french-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleTranslate();
        }
    });

    // Context buttons
    $$('.context-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            handleContextClick(btn.dataset.category);
        });
    });

    // Refresh stats button
    $('#refresh-stats').addEventListener('click', updateSystemStats);

    // Load demo queries
    loadDemoQueries();

    // Initial stats load
    updateSystemStats();

    console.log('LLM-NMT Translation System initialized');
    console.log('Session ID:', sessionId);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
