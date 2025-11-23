// ============================================
// background.js - Service Worker (Lógica Principal)
// ============================================

console.log('🚀 Fake News Detector: Background service worker started');

// Configuración de la API
const API_CONFIG = {
  baseUrl: 'http://localhost:8000',  // Cambiar en producción
  timeout: 30000  // 30 segundos
};

/**
 * Analiza el contenido de una página
 */
async function analyzePage(tabId) {
  try {
    console.log(`📊 Analizando tab ${tabId}...`);
    
    // Verificar si es una página de noticias
    const isNews = await chrome.tabs.sendMessage(tabId, { action: 'isNewsArticle' });
    
    if (!isNews.isNews) {
      console.log('⏭️ No es un artículo de noticias, saltando análisis');
      updateBadge(tabId, '', '#808080'); // Gris
      return;
    }
    
    // Extraer contenido
    const response = await chrome.tabs.sendMessage(tabId, { action: 'extractContent' });
    
    if (!response.success) {
      console.error('❌ Error extrayendo contenido:', response.error);
      updateBadge(tabId, '❌', '#FF0000');
      return;
    }
    
    const content = response.data;
    
    // Validar contenido mínimo
    if (content.wordCount < 50) {
      console.log('⚠️ Contenido muy corto para analizar');
      updateBadge(tabId, '?', '#FFA500');
      return;
    }
    
    // Mostrar badge de "analizando"
    updateBadge(tabId, '...', '#2196F3');
    
    // Enviar a API
    const apiResponse = await fetch(`${API_CONFIG.baseUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: content.text,
        url: content.url
      }),
      signal: AbortSignal.timeout(API_CONFIG.timeout)
    });
    
    if (!apiResponse.ok) {
      throw new Error(`API error: ${apiResponse.status}`);
    }
    
    const result = await apiResponse.json();
    
    console.log('✅ Análisis completado:', result);
    
    // Guardar resultado
    await chrome.storage.local.set({
      [content.url]: {
        ...result,
        title: content.title,
        analyzedAt: new Date().toISOString()
      }
    });
    
    // Actualizar badge con score
    updateBadge(tabId, Math.round(result.score).toString(), getColorForScore(result.score));
    
    return result;
    
  } catch (error) {
    console.error('❌ Error durante análisis:', error);
    updateBadge(tabId, '!', '#FF0000');
    return null;
  }
}

/**
 * Actualiza el badge de la extensión
 */
function updateBadge(tabId, text, color) {
  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
}

/**
 * Retorna color según el score
 */
function getColorForScore(score) {
  if (score >= 75) return '#4CAF50';      // Verde (confiable)
  if (score >= 50) return '#FFC107';      // Amarillo (dudoso)
  return '#F44336';                        // Rojo (fake)
}

// Listener: cuando se carga/actualiza una página
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Ignorar páginas chrome:// y about:
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('about:')) {
      return;
    }
    
    // Analizar página después de 1 segundo (dar tiempo a que cargue contenido)
    setTimeout(() => {
      analyzePage(tabId).catch(err => {
        console.error('Error en análisis automático:', err);
      });
    }, 1000);
  }
});

// Listener: mensajes desde content script o popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'pageLoaded') {
    console.log('📄 Página cargada:', request.url);
  } else if (request.action === 'analyzeManual') {
    // Análisis manual desde popup
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        analyzePage(tabs[0].id).then(sendResponse);
      }
    });
    return true;
  }
});