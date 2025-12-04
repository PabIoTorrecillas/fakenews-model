// ============================================
// popup/popup.js - Lógica del Popup
// ============================================

// Elementos del DOM
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const resultsEl = document.getElementById('results');
const noArticleEl = document.getElementById('no-article');

// Botones
const retryBtn = document.getElementById('retry-btn');
const analyzeAgainBtn = document.getElementById('analyze-again-btn');
const analyzeAnywayBtn = document.getElementById('analyze-anyway-btn');

// Elementos de resultados
const scoreNumber = document.getElementById('score-number');
const scoreCircle = document.getElementById('score-circle');
const classificationBadge = document.getElementById('classification-badge');
const confidence = document.getElementById('confidence');
const probFake = document.getElementById('prob-fake');
const probReal = document.getElementById('prob-real');
const processingTime = document.getElementById('processing-time');
const explanationText = document.getElementById('explanation-text');

/**
 * Muestra una vista específica
 */
function showView(view) {
  [loadingEl, errorEl, resultsEl, noArticleEl].forEach(el => {
    el.classList.add('hidden');
  });
  
  if (view === 'loading') loadingEl.classList.remove('hidden');
  else if (view === 'error') errorEl.classList.remove('hidden');
  else if (view === 'results') resultsEl.classList.remove('hidden');
  else if (view === 'no-article') noArticleEl.classList.remove('hidden');
}

/**
 * Muestra error
 */
function showError(message) {
  document.getElementById('error-message').textContent = message;
  showView('error');
}

/**
 * Carga y muestra resultados
 */
// chrome-extension/popup/popup.js - ARREGLAR LECTURA

async function loadResults() {
  try {
    showView('loading');
    
    // Obtener tab actual
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      showError('No se pudo obtener la página actual');
      return;
    }
    
    console.log('🔍 Buscando análisis para:', tab.url);
    
    // NUEVO: Primero buscar en "latest_analysis"
    const latestResult = await chrome.storage.local.get('latest_analysis');
    
    if (latestResult.latest_analysis && latestResult.latest_analysis.url === tab.url) {
      console.log('✅ Encontrado análisis más reciente');
      displayResults(latestResult.latest_analysis);
      return;
    }
    
    // Fallback: buscar por URL
    const storageKey = `analysis_${tab.url}`;
    const result = await chrome.storage.local.get(storageKey);
    
    if (result[storageKey]) {
      console.log('✅ Encontrado análisis en cache');
      displayResults(result[storageKey]);
    } else {
      console.log('⚠️ No hay análisis previo');
      showView('no-article');
    }
    
  } catch (error) {
    console.error('Error cargando resultados:', error);
    showError('Error al cargar los resultados');
  }
}

function displayResults(data) {
  console.log('📊 Mostrando resultados:', data);
  
  // Score
  const scoreValue = Math.round(data.score);
  scoreNumber.textContent = scoreValue;
  
  console.log(`🎯 Score en UI: ${scoreValue}`);
  
  // Color del círculo según score
  scoreCircle.classList.remove('high', 'medium', 'low');
  if (data.score >= 75) scoreCircle.classList.add('high');
  else if (data.score >= 50) scoreCircle.classList.add('medium');
  else scoreCircle.classList.add('low');
  
  // Clasificación
  const classText = data.classification === 'real' ? '✅ Real' : 
                    data.classification === 'uncertain' ? '⚠️ Dudoso' : '❌ Fake';
  classificationBadge.textContent = classText;
  classificationBadge.classList.remove('real', 'fake', 'uncertain');
  
  if (data.score >= 60) classificationBadge.classList.add('real');
  else if (data.score >= 40) classificationBadge.classList.add('uncertain');
  else classificationBadge.classList.add('fake');
  
  // Detalles
  confidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
  probFake.textContent = `${(data.probabilities.fake * 100).toFixed(1)}%`;
  probReal.textContent = `${(data.probabilities.real * 100).toFixed(1)}%`;
  processingTime.textContent = `${data.processing_time_ms.toFixed(0)}ms`;
  
  console.log('📊 Probabilidades mostradas:');
  console.log(`   Fake: ${(data.probabilities.fake * 100).toFixed(1)}%`);
  console.log(`   Real: ${(data.probabilities.real * 100).toFixed(1)}%`);
  
  // Explicación
  explanationText.textContent = getExplanation(data);
  
  showView('results');
}

// ... resto del código igual ...

/**
 * Genera explicación basada en el resultado
 */
function getExplanation(data) {
  const score = data.score;
  const classification = data.classification;
  
  if (score >= 75) {
    return 'Este artículo tiene características de noticia confiable. El modelo detectó un lenguaje factual y estructura típica de medios verificados.';
  } else if (score >= 50) {
    return 'Este artículo tiene señales mixtas. Se recomienda verificar las fuentes y contrastar con otros medios confiables antes de compartir.';
  } else {
    return 'Este artículo muestra varias características típicas de noticias falsas: lenguaje sensacionalista, falta de fuentes verificables o estructura no periodística. Se recomienda verificar antes de compartir.';
  }
}

/**
 * Solicita nuevo análisis
 */
async function requestAnalysis() {
  try {
    showView('loading');
    
    // Enviar mensaje a background para análisis manual
    chrome.runtime.sendMessage({ action: 'analyzeManual' }, (response) => {
      if (response && response.score !== undefined) {
        displayResults(response);
      } else {
        showError('No se pudo analizar la página. Verifica que sea un artículo de noticias.');
      }
    });
    
  } catch (error) {
    console.error('Error solicitando análisis:', error);
    showError('Error al conectar con el servidor');
  }
}

// Event Listeners
retryBtn.addEventListener('click', loadResults);
analyzeAgainBtn.addEventListener('click', requestAnalysis);
analyzeAnywayBtn.addEventListener('click', requestAnalysis);

// Cargar resultados al abrir popup
loadResults();