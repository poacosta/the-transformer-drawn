const FONT_IMPORT =
  "@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders:opsz,wght@10..72,500;10..72,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');";

const isBoardRule = (cssText: string) =>
  cssText.startsWith(':root') ||
  cssText.startsWith('@keyframes') ||
  cssText.includes('board') ||
  cssText.includes('tb-') ||
  cssText.includes('transformer-board') ||
  cssText.includes('stage-group') ||
  cssText.includes('head-dim');

export function exportSheet(figNumber: number) {
  const svg = document.querySelector<SVGSVGElement>('.transformer-board');
  if (!svg) return;

  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', '1640');
  clone.setAttribute('height', '1080');

  let css = `${FONT_IMPORT}\n`;
  for (const sheet of Array.from(document.styleSheets)) {
    let rules: CSSRuleList;
    try {
      rules = sheet.cssRules;
    } catch {
      continue; // cross-origin stylesheet (e.g. the Google Fonts sheet itself)
    }
    for (const rule of Array.from(rules)) {
      if (isBoardRule(rule.cssText)) {
        css += `${rule.cssText}\n`;
      }
    }
  }

  const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  style.textContent = css;
  clone.insertBefore(style, clone.firstChild);

  const markup = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([markup], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `transformer-fig-${String(figNumber).padStart(2, '0')}.svg`;
  link.click();
  URL.revokeObjectURL(url);
}
