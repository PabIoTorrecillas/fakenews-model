// ============================================
// content.js - Extrae Contenido de la Página
// ============================================

console.log('🔍 Fake News Detector: Content script loaded');

/**
 * Extrae el contenido principal del artículo
 */
function extractArticleContent() {
  // Selectores comunes para artículos de noticias
  const articleSelectors = [
    'article',
    '[role="article"]',
    '.article-content',
    '.post-content',
    '.entry-content',
    '.story-body',
    '.article-body',
    'main article',
    '#article-body',
    '.content-body'
  ];
  
  let articleElement = null;
  
  // Buscar el contenedor del artículo
  for (const selector of articleSelectors) {
    articleElement = document.querySelector(selector);
    if (articleElement && articleElement.innerText.length > 100) {
      break;
    }
  }
  
  // Si no encontramos artículo, usar main o body
  if (!articleElement || articleElement.innerText.length < 100) {
    articleElement = document.querySelector('main') || document.body;
  }
  
  // Extraer texto limpio
  let content = articleElement ? articleElement.innerText : document.body.innerText;
  
  // Limpiar y limitar contenido
  content = content
    .replace(/\s+/g, ' ')  // Normalizar espacios
    .trim()
    .substring(0, 5000);   // Primeros 5000 caracteres
  
  // Extraer metadatos
  const title = document.title || '';
  const url = window.location.href;
  
  // Extraer autor si existe
  const authorMeta = document.querySelector('meta[name="author"]');
  const author = authorMeta ? authorMeta.content : 'Unknown';
  
  return {
    text: content,
    url: url,
    title: title,
    author: author,
    wordCount: content.split(' ').length
  };
}

/**
 * Verifica si la página parece ser un artículo de noticias
 */
function isNewsArticle() {
  const url = window.location.href.toLowerCase();
  const title = document.title.toLowerCase();
  
  // Keywords que indican que es un artículo
  const newsKeywords = [
    'news', 'article', 'story', 'blog', 'post',
    'breaking', 'report', 'noticia', 'articulo'
  ];
  
  // Dominios de noticias conocidos
  const newsDomains = [
    'cnn.com', 'bbc.com', 'nytimes.com', 'reuters.com',
    'theguardian.com', 'washingtonpost.com', 'foxnews.com',
    'nbcnews.com', 'abcnews.com', 'apnews.com'
  ];
  
  // Verificar si es dominio conocido
  const isDomainNews = newsDomains.some(domain => url.includes(domain));
  
  // Verificar keywords
  const hasNewsKeywords = newsKeywords.some(keyword => 
    url.includes(keyword) || title.includes(keyword)
  );
  
  // Verificar longitud del contenido
  const hasEnoughContent = document.body.innerText.length > 500;
  
  return (isDomainNews || hasNewsKeywords) && hasEnoughContent;
}

// Escuchar mensajes desde background.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'extractContent') {
    try {
      const content = extractArticleContent();
      sendResponse({ success: true, data: content });
    } catch (error) {
      sendResponse({ success: false, error: error.message });
    }
  } else if (request.action === 'isNewsArticle') {
    sendResponse({ isNews: isNewsArticle() });
  }
  
  return true; // Mantener el canal abierto para respuesta asíncrona
});

// Notificar que la página está lista
if (document.readyState === 'complete') {
  chrome.runtime.sendMessage({ action: 'pageLoaded', url: window.location.href });
} else {
  window.addEventListener('load', () => {
    chrome.runtime.sendMessage({ action: 'pageLoaded', url: window.location.href });
  });
}