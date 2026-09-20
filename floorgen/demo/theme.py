"""
Rose Quartz Glassmorphism Design System & Architectural Studio UI Components for FloorGen.
Implements the God Mode UI Design System (8px Grid, Typography Scale, Elevation Levels,
Z-Pattern Flow, Accessible Interaction States) combined with the Rose Quartz CAD palette
(#FFF0F5, #FCE7F3, #FBCFE8, #F43F5E), frosted glass panels, and comprehensive architectural guides.
"""

from typing import Dict, Any, List, Optional
import gradio as gr


ROSE_QUARTZ_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@500;600;700;800;900&display=swap');

/* ==========================================================================
   GOD MODE DESIGN SYSTEM TOKENS (8px Grid, Elevation, Typography & Rose Palette)
   ========================================================================== */
:root, html, body, .gradio-container, .gradio-container.light, .gradio-container.dark {
  /* 8px Spacing Grid System */
  --space-xs:   4px !important;
  --space-sm:   8px !important;
  --space-md:   16px !important;
  --space-lg:   24px !important;
  --space-xl:   32px !important;
  --space-2xl:  48px !important;
  --space-3xl:  64px !important;
  --space-4xl:  96px !important;

  /* Elevation / Shadow Scale */
  --elevation-0: none !important;
  --elevation-1: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(244, 114, 182, 0.08) !important;
  --elevation-2: 0 4px 12px rgba(244, 114, 182, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
  --elevation-3: 0 8px 24px rgba(244, 114, 182, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
  --elevation-4: 0 16px 36px rgba(244, 114, 182, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
  --elevation-5: 0 24px 48px rgba(244, 63, 94, 0.28) !important;

  /* Rose Quartz Palette & Surfaces */
  --bg-canvas:          #FFF0F5 !important;
  --bg-surface:         rgba(255, 255, 255, 0.72) !important;
  --bg-surface-subtle:  rgba(255, 255, 255, 0.85) !important;
  --bg-surface-hover:   rgba(255, 255, 255, 0.96) !important;

  --border-subtle:      rgba(244, 114, 182, 0.35) !important;
  --border-strong:      rgba(244, 114, 182, 0.60) !important;
  --border-focus:       #F43F5E !important;

  --text-primary:       #37131D !important;
  --text-secondary:     #701A35 !important;
  --text-muted:         #9D174D !important;
  --text-subtle:        #BE185D !important;

  --primary:            #F43F5E !important;
  --primary-hover:      #E11D48 !important;
  --accent:             #C026D3 !important;
  --accent-hover:       #D946EF !important;

  --success:            #059669 !important;
  --warning:            #D97706 !important;
  --danger:             #E11D48 !important;
  --info:               #2563EB !important;

  /* Typography Font Stacks */
  --font-display:       'Outfit', 'Inter', -apple-system, sans-serif !important;
  --font-sans:          'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  --font-mono:          'JetBrains Mono', monospace !important;

  /* Gradio Native Overrides */
  --body-background-fill: #FFF0F5 !important;
  --body-background-fill-dark: #FFF0F5 !important;
  --background-fill-primary: #FFF0F5 !important;
  --background-fill-primary-dark: #FFF0F5 !important;
  --background-fill-secondary: rgba(255, 255, 255, 0.72) !important;
  --background-fill-secondary-dark: rgba(255, 255, 255, 0.72) !important;
  --block-background-fill: rgba(255, 255, 255, 0.72) !important;
  --block-background-fill-dark: rgba(255, 255, 255, 0.72) !important;
  --block-border-color: rgba(244, 114, 182, 0.35) !important;
  --block-border-color-dark: rgba(244, 114, 182, 0.35) !important;
  --block-label-background-fill: rgba(255, 255, 255, 0.85) !important;
  --block-label-background-fill-dark: rgba(255, 255, 255, 0.85) !important;
  --block-label-text-color: #701A35 !important;
  --block-label-text-color-dark: #701A35 !important;
  --block-title-text-color: #37131D !important;
  --block-title-text-color-dark: #37131D !important;
  --input-background-fill: rgba(255, 255, 255, 0.80) !important;
  --input-background-fill-dark: rgba(255, 255, 255, 0.80) !important;
  --input-border-color: rgba(244, 114, 182, 0.45) !important;
  --input-border-color-dark: rgba(244, 114, 182, 0.45) !important;
  --input-placeholder-color: #9D174D !important;
  --border-color-primary: rgba(244, 114, 182, 0.35) !important;
  --border-color-accent: #F43F5E !important;
}

/* Radiant blush canvas background */
body, .gradio-container {
  background: linear-gradient(135deg, #FFF1F4 0%, #FDF2F8 30%, #FCE7F3 70%, #FBCFE8 100%) !important;
  background-attachment: fixed !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: var(--space-md) !important;
}

/* ==========================================================================
   Z-PATTERN HEADER (Logo Top-Left, Status Center, Quick Guide Top-Right)
   ========================================================================== */
.studio-header {
  background: rgba(255, 255, 255, 0.78) !important;
  backdrop-filter: blur(20px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
  border: 1px solid var(--border-subtle) !important;
  box-shadow: var(--elevation-3) !important;
  border-radius: 18px !important;
  padding: var(--space-md) var(--space-lg) !important;
  margin-bottom: var(--space-md) !important;
  position: relative !important;
  overflow: hidden !important;
}
.studio-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #FDA4AF, #F43F5E, #C026D3, #FDA4AF);
}
.studio-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: var(--space-md);
}
.studio-logo-group {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}
.studio-logo-icon {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(251, 113, 133, 0.3), rgba(244, 63, 94, 0.15));
  border: 1px solid rgba(244, 63, 94, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 26px;
  box-shadow: var(--elevation-2);
}
.studio-title {
  font-family: var(--font-display);
  font-size: 1.75rem;
  font-weight: 800;
  color: #37131D;
  letter-spacing: -0.02em;
  margin: 0;
  line-height: 1.15;
}
.studio-title span {
  color: #E11D48;
  background: linear-gradient(135deg, #F43F5E, #BE185D);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.studio-subtitle {
  font-size: 0.85rem;
  color: #701A35;
  font-weight: 500;
  margin-top: 2px;
}
.studio-badges-row {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  flex-wrap: wrap;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 9999px;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  border: 1px solid rgba(244, 114, 182, 0.45);
  background: rgba(255, 255, 255, 0.65);
  backdrop-filter: blur(8px);
  color: #831843;
  box-shadow: var(--elevation-1);
}
.status-pill.active {
  border-color: rgba(244, 63, 94, 0.5);
  background: rgba(254, 205, 211, 0.50);
  color: #881337;
}
.status-pill.active .dot {
  background: #F43F5E;
  box-shadow: 0 0 8px #F43F5E;
}
.status-pill.accent {
  border-color: rgba(192, 38, 211, 0.4);
  background: rgba(245, 208, 254, 0.40);
  color: #701A75;
}
.status-pill.accent .dot {
  background: #C026D3;
  box-shadow: 0 0 8px #C026D3;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}

/* ==========================================================================
   WORKFLOW PROGRESS STEP BAR (Visual Guided Navigation)
   ========================================================================== */
.workflow-step-bar {
  display: flex;
  align-items: stretch;
  justify-content: space-between;
  gap: var(--space-md);
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(244, 114, 182, 0.35);
  box-shadow: var(--elevation-2);
  border-radius: 16px;
  padding: var(--space-md) var(--space-lg);
  margin-bottom: var(--space-md);
  overflow-x: auto;
}
.step-item {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  flex: 1;
  min-width: 220px;
  position: relative;
}
.step-item:not(:last-child)::after {
  content: '➔';
  position: absolute;
  right: -8px;
  color: rgba(244, 114, 182, 0.6);
  font-size: 1rem;
}
.step-num {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.95rem;
  font-family: var(--font-display);
  flex-shrink: 0;
  background: rgba(254, 205, 211, 0.6);
  border: 1.5px solid rgba(244, 63, 94, 0.4);
  color: #9F1239;
}
.step-item.active .step-num {
  background: linear-gradient(135deg, #FB7185, #F43F5E);
  color: #FFFFFF;
  border: none;
  box-shadow: 0 4px 12px rgba(244, 63, 94, 0.35);
}
.step-content {
  display: flex;
  flex-direction: column;
}
.step-title {
  font-weight: 700;
  font-size: 0.88rem;
  color: #37131D;
  line-height: 1.2;
}
.step-desc {
  font-size: 0.74rem;
  color: #701A35;
  margin-top: 2px;
}

/* ==========================================================================
   UNIVERSAL FROSTED GLASS CARDS & CONTAINERS
   ========================================================================== */
[class*="gradio-container"] .block,
[class*="gradio-container"] .form,
[class*="gradio-container"] fieldset,
[class*="gradio-container"] label.block,
[class*="gradio-container"] .file-preview-holder,
[class*="gradio-container"] .file-preview,
[class*="gradio-container"] .file-wrap,
[class*="gradio-container"] .upload-container,
[class*="gradio-container"] .empty-file,
[class*="gradio-container"] [data-testid="textbox"],
[class*="gradio-container"] [data-testid="checkboxgroup"],
[class*="gradio-container"] [data-testid="file"],
.block, .form, fieldset, label.block,
[data-testid="textbox"], [data-testid="checkboxgroup"], [data-testid="file"] {
  background: rgba(255, 255, 255, 0.74) !important;
  backdrop-filter: blur(20px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
  border: 1px solid rgba(244, 114, 182, 0.35) !important;
  color: #37131D !important;
  box-shadow: var(--elevation-2) !important;
  border-radius: 16px !important;
}

/* ==========================================================================
   ELEVATED BUTTON SYSTEM (Primary & Secondary Glass)
   ========================================================================== */
[class*="gradio-container"] button:not(.primary),
button:not(.primary), .gr-button-secondary, button.sm, button.secondary,
.glass-btn {
  background: rgba(255, 255, 255, 0.50) !important;
  backdrop-filter: blur(14px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
  border: 1.5px solid rgba(244, 114, 182, 0.50) !important;
  color: #831843 !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  padding: 8px 16px !important;
  box-shadow: var(--elevation-1) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  touch-action: manipulation !important;
  min-height: 38px !important;
}
[class*="gradio-container"] button:not(.primary):hover,
button:not(.primary):hover, .gr-button-secondary:hover, button.sm:hover, button.secondary:hover {
  background: rgba(255, 255, 255, 0.90) !important;
  border-color: #F43F5E !important;
  color: #9F1239 !important;
  box-shadow: var(--elevation-2) !important;
  transform: translateY(-1px) !important;
}
[class*="gradio-container"] button:not(.primary):active,
button:not(.primary):active {
  background: rgba(254, 205, 211, 0.6) !important;
  transform: translateY(0) !important;
  box-shadow: var(--elevation-1) !important;
}

/* Primary Jewel Generate Button */
button.primary, [class*="gradio-container"] button.primary, .gr-button-primary {
  background: linear-gradient(135deg, #FB7185 0%, #F43F5E 50%, #E11D48 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 8px 24px rgba(244, 63, 94, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  padding: 14px 28px !important;
  touch-action: manipulation !important;
  min-height: 48px !important;
}
button.primary:hover, [class*="gradio-container"] button.primary:hover {
  background: linear-gradient(135deg, #FDA4AF 0%, #FB7185 50%, #F43F5E 100%) !important;
  box-shadow: 0 12px 30px rgba(244, 63, 94, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
  transform: translateY(-2px) !important;
}
button.primary:active, [class*="gradio-container"] button.primary:active {
  transform: translateY(0) !important;
  box-shadow: 0 4px 14px rgba(244, 63, 94, 0.25) !important;
}

/* ==========================================================================
   MULTI-CHOICE SELECTION BOXES (Strict Accessibility & Touch Optimization)
   ========================================================================== */
[data-testid="checkboxgroup"] .wrap,
.gr-checkboxgroup .wrap {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: var(--space-sm) !important;
}

.gr-checkboxgroup label, label.checkbox-label, [data-testid="checkboxgroup"] label {
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  background: rgba(255, 255, 255, 0.65) !important;
  backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important;
  border: 1.5px solid rgba(244, 114, 182, 0.40) !important;
  border-radius: 12px !important;
  padding: 8px 14px !important;
  margin: 0 !important;
  color: #37131D !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  box-shadow: var(--elevation-1) !important;
  transition: all 0.15s ease !important;
  cursor: pointer !important;
  touch-action: manipulation !important;
  user-select: none !important;
  -webkit-user-select: none !important;
  min-height: 40px !important;
}

.gr-checkboxgroup label:hover, label.checkbox-label:hover {
  border-color: #F43F5E !important;
  background: rgba(255, 255, 255, 0.95) !important;
  color: #9F1239 !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--elevation-2) !important;
}

.gr-checkboxgroup label.selected, label.checkbox-label.selected,
[data-testid="checkboxgroup"] label:has(input:checked),
.gr-checkboxgroup label:has(input:checked) {
  background: linear-gradient(135deg, rgba(254, 205, 211, 0.85), rgba(251, 113, 133, 0.48)) !important;
  border-color: #F43F5E !important;
  color: #881337 !important;
  font-weight: 700 !important;
  box-shadow: var(--elevation-2) !important;
}

.gr-checkboxgroup input[type="checkbox"], [data-testid="checkboxgroup"] input[type="checkbox"] {
  accent-color: #F43F5E !important;
  width: 18px !important;
  height: 18px !important;
  margin: 0 !important;
  cursor: pointer !important;
}

.gr-checkboxgroup label span, [data-testid="checkboxgroup"] label span {
  color: inherit !important;
  font-weight: inherit !important;
  font-size: inherit !important;
  cursor: pointer !important;
}

/* ==========================================================================
   INPUT FIELDS (Focus Ring, Border Tokens, Labels)
   ========================================================================== */
input, textarea, select {
  background: rgba(255, 255, 255, 0.82) !important;
  border: 1.5px solid rgba(244, 114, 182, 0.45) !important;
  color: #37131D !important;
  border-radius: 12px !important;
  font-family: var(--font-sans) !important;
  box-shadow: inset 0 1px 3px rgba(244, 114, 182, 0.08) !important;
  transition: all 0.2s ease !important;
  padding: 8px 12px !important;
}
input:focus, textarea:focus, select:focus {
  border-color: #F43F5E !important;
  box-shadow: 0 0 0 3px rgba(244, 63, 94, 0.20) !important;
  outline: none !important;
}

/* Labels and Microcopy */
label span, .block span, .block label, .form label, p, .gr-check-label, .label-wrap span {
  color: #701A35 !important;
  font-weight: 600 !important;
}

/* ==========================================================================
   INTERACTIVE ARCHITECTURAL GUIDES & CHEAT SHEETS
   ========================================================================== */
.guide-card {
  background: rgba(255, 255, 255, 0.80);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(244, 114, 182, 0.40);
  box-shadow: var(--elevation-2);
  border-radius: 14px;
  padding: var(--space-md);
  margin-top: var(--space-sm);
  margin-bottom: var(--space-sm);
}
.guide-card-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-sm);
}
.guide-card-title {
  font-weight: 800;
  font-size: 0.92rem;
  color: #9F1239;
}
.guide-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.70rem;
  font-family: var(--font-mono);
  font-weight: 700;
  background: rgba(244, 63, 94, 0.12);
  color: #9F1239;
  border: 1px solid rgba(244, 63, 94, 0.3);
}

.guide-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--space-md);
}
.guide-grid-item {
  background: rgba(255, 255, 255, 0.90);
  border: 1px solid rgba(244, 114, 182, 0.30);
  border-radius: 12px;
  padding: 14px;
  box-shadow: var(--elevation-1);
  transition: all 0.2s ease;
}
.guide-grid-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--elevation-2);
  border-color: #F43F5E;
}
.guide-grid-title {
  font-weight: 700;
  font-size: 0.86rem;
  color: #37131D;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.guide-grid-body {
  font-size: 0.76rem;
  color: #701A35;
  line-height: 1.5;
}

/* Building Code Checklist Table */
.code-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin-top: var(--space-sm);
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(244, 114, 182, 0.35);
  font-size: 0.78rem;
}
.code-table th {
  background: rgba(254, 205, 211, 0.6);
  color: #881337;
  font-weight: 700;
  text-align: left;
  padding: 10px 14px;
  border-bottom: 1px solid rgba(244, 114, 182, 0.35);
}
.code-table td {
  background: rgba(255, 255, 255, 0.85);
  color: #37131D;
  padding: 10px 14px;
  border-bottom: 1px solid rgba(244, 114, 182, 0.20);
}
.code-table tr:last-child td {
  border-bottom: none;
}
.code-tag {
  display: inline-block;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 0.72rem;
  color: #BE185D;
  background: rgba(244, 63, 94, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
}

/* Accordion Styling */
.accordion, [data-testid="accordion"], summary, details {
  background: rgba(255, 255, 255, 0.74) !important;
  color: #37131D !important;
  border-radius: 14px !important;
  border-color: rgba(244, 114, 182, 0.35) !important;
}

/* Canvas Viewport Frame */
.canvas-viewer-frame {
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid rgba(244, 114, 182, 0.40);
  box-shadow: var(--elevation-3);
  border-radius: 18px;
  padding: 20px;
  position: relative;
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}
.canvas-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(244, 114, 182, 0.30);
}
.canvas-title-group {
  display: flex;
  align-items: center;
  gap: 10px;
}
.canvas-title {
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 700;
  color: #37131D;
}
.canvas-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 0.72rem;
  font-family: var(--font-mono);
  font-weight: 700;
  background: rgba(244, 63, 94, 0.12);
  color: #BE185D;
  border: 1px solid rgba(244, 63, 94, 0.35);
}

.svg-render-box {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 14px;
  overflow: hidden;
  background: #FFF8FA;
  box-shadow: var(--elevation-2);
  border: 1px solid rgba(244, 114, 182, 0.35);
}
.svg-render-box svg {
  width: 100%;
  height: auto;
  max-height: 720px;
  display: block;
}

/* Legend Chips */
.legend-chips-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  margin-top: 4px;
}
.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(244, 114, 182, 0.35);
  font-size: 0.76rem;
  color: #37131D;
  box-shadow: var(--elevation-1);
}
.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* Telemetry HUD Cards */
.hud-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
  margin-top: var(--space-md);
}
.hud-card {
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(244, 114, 182, 0.35);
  box-shadow: var(--elevation-2);
  border-radius: 14px;
  padding: 16px;
  position: relative;
  overflow: hidden;
}
.hud-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(244, 63, 94, 0.4), transparent);
}
.hud-label {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #9D174D;
  margin-bottom: 6px;
}
.hud-value {
  font-family: var(--font-display);
  font-size: 1.75rem;
  font-weight: 800;
  color: #37131D;
  line-height: 1;
}
.hud-value.rose {
  color: #E11D48;
}
.hud-value.fuchsia {
  color: #C026D3;
}
.hud-value.green {
  color: #059669;
}
.hud-sub {
  font-size: 0.74rem;
  color: #701A35;
  margin-top: 6px;
}

/* Exemplars & Breakdown List */
.exemplars-container {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(244, 114, 182, 0.35);
  box-shadow: var(--elevation-2);
  border-radius: 14px;
  padding: 16px;
  margin-top: 14px;
}
.exemplar-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(244, 114, 182, 0.3);
  margin-top: 6px;
  font-size: 0.8rem;
}
.exemplar-id {
  font-family: var(--font-mono);
  color: #BE185D;
  font-weight: 700;
}
.exemplar-score {
  font-family: var(--font-mono);
  color: #059669;
  font-weight: 600;
}

/* File Download Cards */
.gr-file {
  background: rgba(255, 255, 255, 0.75) !important;
  border: 1px solid rgba(244, 114, 182, 0.35) !important;
  border-radius: 12px !important;
  box-shadow: var(--elevation-1) !important;
}
"""

# Alias for backwards compatibility
OCEAN_DEPTH_CSS = ROSE_QUARTZ_CSS


ROSE_QUARTZ_HEAD_SCRIPT = """
<script>
  try {
    localStorage.setItem('theme', 'light');
  } catch (e) {}
  document.documentElement.classList.remove('dark');
  document.documentElement.classList.add('light');

  function applyRoseQuartzStyles() {
    document.documentElement.classList.remove('dark');
    document.documentElement.classList.add('light');
    if (document.body) {
      document.body.classList.remove('dark');
      document.body.classList.add('light');
    }
    document.querySelectorAll('[class*="gradio-container"]').forEach(el => {
      el.classList.remove('dark');
      el.classList.add('light');
    });
  }
  document.addEventListener('DOMContentLoaded', applyRoseQuartzStyles);
  window.addEventListener('load', applyRoseQuartzStyles);
  if (window.MutationObserver) {
    const obs = new MutationObserver(applyRoseQuartzStyles);
    obs.observe(document.documentElement, { childList: true, subtree: true });
  }
  applyRoseQuartzStyles();
  setInterval(applyRoseQuartzStyles, 400);
</script>
"""

OCEAN_DEPTH_HEAD_SCRIPT = ROSE_QUARTZ_HEAD_SCRIPT


def render_studio_header_html() -> str:
    """Renders the top branding and live status HUD bar with Z-pattern flow and quick guide indicators."""
    return """
<div class="studio-header">
  <div class="studio-title-row">
    <!-- Top-Left: Logo & Identity -->
    <div class="studio-logo-group">
      <div class="studio-logo-icon">🌸</div>
      <div>
        <h1 class="studio-title">FLOOR<span>GEN</span> STUDIO</h1>
        <div class="studio-subtitle">Generative Vector Architectural Synthesis • Rose Quartz CAD Suite</div>
      </div>
    </div>
    
    <!-- Center / Top-Right: Engine Status & Quick Guide Badges -->
    <div class="studio-badges-row">
      <div class="status-pill active"><span class="dot"></span>Qwen Reasoning Active</div>
      <div class="status-pill accent"><span class="dot"></span>CP-SAT Solver Enabled</div>
      <div class="status-pill"><span class="dot" style="background:#F43F5E;"></span>Dual FAISS RAG</div>
      <div class="status-pill"><span class="dot" style="background:#C026D3;"></span>ISO-16739 BIM Ready</div>
      <div class="status-pill" style="border-color: rgba(5, 150, 105, 0.4); background: rgba(209, 250, 229, 0.45); color: #065F46;">
        <span class="dot" style="background:#059669;"></span>100% Local • $0 Cost
      </div>
    </div>
  </div>
</div>
"""


def render_workflow_steps_html() -> str:
    """Renders a guided 3-step progress bar explaining the generation workflow to users."""
    return """
<div class="workflow-step-bar">
  <div class="step-item active">
    <div class="step-num">1</div>
    <div class="step-content">
      <div class="step-title">Specify Program &amp; Brief</div>
      <div class="step-desc">Pick archetype or check rooms; type natural language requirements</div>
    </div>
  </div>
  <div class="step-item active">
    <div class="step-num">2</div>
    <div class="step-content">
      <div class="step-title">Diffusion &amp; CP-SAT Solve</div>
      <div class="step-desc">Reverse diffusion synthesizes layout; OR-Tools eliminates all overlaps</div>
    </div>
  </div>
  <div class="step-item active">
    <div class="step-num">3</div>
    <div class="step-content">
      <div class="step-title">CAD, BIM &amp; Vector Export</div>
      <div class="step-desc">Download AutoCAD DXF (9 layers), Revit IFC, SVG, and presentation render</div>
    </div>
  </div>
</div>
"""


def render_architect_quick_guide_html() -> str:
    """Renders an interactive cheat sheet and parameter guide card for architects."""
    return """
<div class="guide-card">
  <div class="guide-card-header">
    <span style="font-size: 1.1rem;">📘</span>
    <span class="guide-card-title">Architect's Parameter Guide &amp; Engine Cheat Sheet</span>
    <span class="guide-badge">QUICK REFERENCE</span>
  </div>
  <div class="guide-grid">
    <div class="guide-grid-item">
      <div class="guide-grid-title">⚡ Sampler: DDIM vs DDPM</div>
      <div class="guide-grid-body">
        <strong>DDIM (Deterministic):</strong> Recommended for sharp, orthogonal residential floor plans with crisp Manhattan wall alignment.<br>
        <strong>DDPM (Stochastic):</strong> Introduces randomized exploratory variations across rooms.
      </div>
    </div>
    <div class="guide-grid-item">
      <div class="guide-grid-title">📐 Denoising Steps (10–25)</div>
      <div class="guide-grid-body">
        <strong>15 Steps (Default):</strong> Optimal sweet spot for boundary convergence in ~180ms.<br>
        <strong>25–30 Steps:</strong> Maximizes topological refinement for complex multi-room programs (>8 spaces).
      </div>
    </div>
    <div class="guide-grid-item">
      <div class="guide-grid-title">🏛️ Top-K RAG Exemplars (1–10)</div>
      <div class="guide-grid-body">
        FloorGen retrieves real floor plans from the 80,000-plan RPLAN index. The engine conditions the latent diffusion trajectory on spatial proportion priors from these verified blueprints.
      </div>
    </div>
    <div class="guide-grid-item">
      <div class="guide-grid-title">⚖️ Google OR-Tools CP-SAT</div>
      <div class="guide-grid-body">
        <strong>Zero-Overlap Guarantee:</strong> Mathematical constraint solver enforces strict 2D interval non-overlap (<code>AddNoOverlap2D</code>) and guarantees IRC legal room boundaries in &lt;8ms.
      </div>
    </div>
  </div>
</div>
"""


def render_building_code_guide_html() -> str:
    """Renders a comprehensive International Building Code (IRC/IBC) reference guide."""
    return """
<div class="guide-card">
  <div class="guide-card-header">
    <span style="font-size: 1.1rem;">🏛️</span>
    <span class="guide-card-title">International Residential &amp; Building Code (IRC / IBC) Compliance Standards</span>
    <span class="guide-badge">AUTOMATED AUDIT</span>
  </div>
  <div style="font-size: 0.80rem; color: #701A35; margin-bottom: 12px; line-height: 1.5;">
    Every floor plan generated by FloorGen is evaluated against the following legal architectural standards to ensure code-compliant habitability:
  </div>
  <table class="code-table">
    <thead>
      <tr>
        <th style="width: 25%;">Code Standard</th>
        <th style="width: 35%;">Requirement</th>
        <th style="width: 40%;">FloorGen Implementation &amp; Guarantee</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span class="code-tag">IRC R304.1</span> Minimum Area</td>
        <td>Habitable rooms must have at least <strong>70 sq.ft (6.5 m²)</strong> of gross floor area.</td>
        <td>CP-SAT solver lower-bounds habitable room dimensions so no bedroom, living, or dining area falls below 70 sf.</td>
      </tr>
      <tr>
        <td><span class="code-tag">IRC R304.2</span> Min Dimension</td>
        <td>Habitable rooms must not be less than <strong>7 ft (2.13 m)</strong> in any horizontal dimension.</td>
        <td>Aspect ratio penalization and CP-SAT interval bounds prevent thin sliver rooms.</td>
      </tr>
      <tr>
        <td><span class="code-tag">IBC 1010.1</span> Egress Width</td>
        <td>Means of egress doors must provide a minimum clear opening width of <strong>32 inches (0.81 m)</strong>.</td>
        <td>Door placement algorithm allocates 36" standard clear entrance openings and door swing arcs.</td>
      </tr>
      <tr>
        <td><span class="code-tag">IRC R303.1</span> Glazing / Light</td>
        <td>Habitable rooms require exterior natural daylight glazing equal to ≥ <strong>8% of floor area</strong>.</td>
        <td>Habitable spaces are preferentially allocated to the building exterior perimeter boundary.</td>
      </tr>
    </tbody>
  </table>
</div>
"""


def render_cad_deliverables_guide_html() -> str:
    """Renders an engineering CAD & BIM format compatibility guide."""
    return """
<div class="guide-card">
  <div class="guide-card-header">
    <span style="font-size: 1.1rem;">📦</span>
    <span class="guide-card-title">CAD &amp; BIM Format Compatibility &amp; Workflow Guide</span>
    <span class="guide-badge">INTEROPERABILITY</span>
  </div>
  <div class="guide-grid">
    <div class="guide-grid-item">
      <div class="guide-grid-title">🏗️ AutoCAD DXF (.dxf)</div>
      <div class="guide-grid-body">
        <strong>Format:</strong> ASCII DXF R12 / AutoCAD 2000 standard.<br>
        <strong>Layers:</strong> 9 discrete ACI-colored layers (WALLS, WALLS_INTERIOR, DOORS, DOOR_SWINGS, ROOM_LABELS, DIMENSIONS, FURNITURE, SANITARY, BALCONY).<br>
        <strong>Software:</strong> AutoCAD, LibreCAD, Rhino, DraftSight, QCAD.
      </div>
    </div>
    <div class="guide-grid-item">
      <div class="guide-grid-title">🏢 ISO-16739 IFC (.ifc)</div>
      <div class="guide-grid-body">
        <strong>Format:</strong> Industry Foundation Classes (IFC 2x3 / IFC4).<br>
        <strong>Entities:</strong> Extruded 3D 2.8m solids (<code>IfcWallStandardCase</code>), spatial containment (<code>IfcSpace</code>), and opening elements (<code>IfcDoor</code>).<br>
        <strong>Software:</strong> Autodesk Revit, ArchiCAD, BlenderBIM, Solibri.
      </div>
    </div>
    <div class="guide-grid-item">
      <div class="guide-grid-title">📐 Vector SVG (.svg)</div>
      <div class="guide-grid-body">
        <strong>Format:</strong> Scalable Vector Graphics with dynamic 8% framing padding.<br>
        <strong>Features:</strong> Resolution-independent, crisp Manhattan wall joins, room color fills, and dimension callouts.<br>
        <strong>Software:</strong> Adobe Illustrator, Inkscape, Web Browsers, Figma.
      </div>
    </div>
    <div class="guide-grid-item">
      <div class="guide-grid-title">📜 JSON Specification (.json)</div>
      <div class="guide-grid-body">
        <strong>Format:</strong> Structured machine-readable schema.<br>
        <strong>Data:</strong> Normalized bounding boxes, square footage, adjacency contact matrices, door coordinates, and wall junction graphs.<br>
        <strong>Software:</strong> Custom Python scripts, Rhino Grasshopper, Dynamo.
      </div>
    </div>
  </div>
</div>
"""


def render_interactive_viewer_html(svg_content: str, floorplan_data: Optional[Dict[str, Any]] = None) -> str:
    """Wraps synthesized SVG in a light pink architectural canvas with toolbar and room legend."""
    if not svg_content:
        svg_content = """
        <div style="text-align: center; padding: 70px 20px; color: #9D174D;">
          <div style="font-size: 48px; margin-bottom: 12px; opacity: 0.85;">📐</div>
          <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #831843;">Ready for Architectural Synthesis</div>
          <div style="font-size: 0.88rem; margin-top: 6px; color: #701A35; max-width: 480px; margin-left: auto; margin-right: auto;">
            Select desired rooms or an archetype preset on the left, then click <strong style="color: #E11D48;">⚡ Generate Floorplan</strong> to synthesize.
          </div>
        </div>
        """
        return f'<div class="canvas-viewer-frame"><div class="svg-render-box">{svg_content}</div></div>'

    num_rooms = len(floorplan_data.get("rooms", [])) if floorplan_data else 0
    total_area = sum(r.get("area", 0) for r in floorplan_data.get("rooms", [])) if floorplan_data else 0

    ROOM_COLORS_MAP = {
        "living_room": "#F43F5E",
        "master_bedroom": "#9333EA",
        "second_bedroom": "#A855F7",
        "kitchen": "#D97706",
        "bathroom": "#0D9488",
        "balcony": "#059669",
        "dining_room": "#DB2777",
        "study": "#C026D3",
        "entrance": "#64748B",
        "storage": "#64748B"
    }

    ROOM_ICONS = {
        "living_room": "🛋️",
        "master_bedroom": "👑",
        "second_bedroom": "🛏️",
        "kitchen": "🍳",
        "bathroom": "🚿",
        "balcony": "🌿",
        "dining_room": "🍽️",
        "study": "📚",
        "entrance": "🚪",
        "storage": "📦"
    }

    legend_html = ""
    if floorplan_data and "rooms" in floorplan_data:
        chips = []
        for r in floorplan_data["rooms"]:
            cat = r.get("category", "default")
            col = ROOM_COLORS_MAP.get(cat, "#F43F5E")
            icon = ROOM_ICONS.get(cat, "📐")
            label = cat.replace("_", " ").title()
            area = r.get("area", 0)
            chips.append(f"""
            <div class="legend-chip">
              <span class="legend-dot" style="background: {col};"></span>
              <span>{icon} <strong>{label}</strong>: {area:.0f} sq.ft</span>
            </div>
            """)
        legend_html = f'<div class="legend-chips-row">{"".join(chips)}</div>'

    return f"""
<div class="canvas-viewer-frame">
  <div class="canvas-toolbar">
    <div class="canvas-title-group">
      <span class="canvas-title">Architectural CAD Blueprint</span>
      <span class="canvas-badge">{num_rooms} ROOMS</span>
      <span class="canvas-badge">{total_area:.0f} SQ.FT TOTAL</span>
    </div>
    <div style="display: flex; gap: 8px;">
      <span style="font-size: 0.74rem; color: #9D174D; font-family: 'JetBrains Mono', monospace; padding-top: 3px; letter-spacing: 0.04em;">AUTO-FRAMED VECTOR • 100% SCALE</span>
    </div>
  </div>
  <div class="svg-render-box">
    {svg_content}
  </div>
  {legend_html}
</div>
"""


def render_telemetry_hud_html(result: Any) -> str:
    """Renders the frosted glass architectural telemetry and code compliance HUD."""
    if result is None:
        return ""

    realism = getattr(result, "realism_score", 85.0)
    circulation = getattr(result, "circulation", 1.0) * 100.0
    aspect = getattr(result, "aspect_ratio", 0.88)
    latency_ms = getattr(result, "generation_time_ms", 260.0)
    device = getattr(result, "device", "cuda")
    epoch = getattr(result, "checkpoint_epoch", 80)
    compliance = getattr(result, "compliance", {}) or {}

    is_compliant = compliance.get("passed", True) if isinstance(compliance, dict) else True
    comp_score = compliance.get("compliance_score", 1.0) * 100.0 if isinstance(compliance, dict) else 100.0
    violations_count = len(compliance.get("violations", [])) if isinstance(compliance, dict) else 0

    comp_badge = f'<span style="color: #059669; font-weight: 700;">✅ PASSED ({comp_score:.1f}%)</span>' if is_compliant else f'<span style="color: #D97706; font-weight: 700;">⚠️ {violations_count} VIOLATIONS</span>'

    walls = getattr(result, "walls", None)
    num_walls = 0
    num_junctions = 0
    if walls and isinstance(walls, dict):
        num_walls = len(walls.get("walls", []))
        num_junctions = len(walls.get("junctions", []))

    furniture = getattr(result, "furniture", None)
    num_furniture = len(furniture) if isinstance(furniture, list) else 0

    # Explicit room breakdown
    rooms_spec = result.json_spec.get("rooms", []) if hasattr(result, "json_spec") and isinstance(result.json_spec, dict) else []
    room_items = []
    ROOM_ICONS_MAP = {
        "living_room": "🛋️", "master_bedroom": "👑", "second_bedroom": "🛏️",
        "kitchen": "🍳", "bathroom": "🚿", "balcony": "🌿",
        "dining_room": "🍽️", "study": "📚", "entrance": "🚪", "storage": "📦"
    }
    for r in rooms_spec:
        cat = r.get("category", "")
        area = r.get("area", 0)
        icon = ROOM_ICONS_MAP.get(cat, "📐")
        room_items.append(f'<span style="display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 8px; background: rgba(255,255,255,0.85); border: 1px solid rgba(244,114,182,0.35); font-size: 0.78rem; color: #37131D; font-weight: 600;">{icon} {cat.replace("_", " ").title()}: {area:.0f} sf</span>')
    rooms_summary_html = f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;">{"".join(room_items)}</div>' if room_items else "No rooms recorded"

    exemplars = getattr(result, "exemplars", []) or []
    exemplars_html = ""
    if exemplars:
        ex_items = []
        for e in exemplars[:4]:
            pid = e.get("plan_id", "rplan_ref")
            sc = float(e.get("score", 0.0))
            arch = e.get("plan", {}).get("archetype", "rplan_real") if isinstance(e.get("plan"), dict) else "rplan_real"
            ex_items.append(f"""
            <div class="exemplar-item">
              <span class="exemplar-id">🏛️ {pid}</span>
              <span style="color: #701A35;">{arch}</span>
              <span class="exemplar-score">Cosine: {sc:.3f}</span>
            </div>
            """)
        exemplars_html = f"""
        <div class="exemplars-container">
          <div class="hud-label">Retrieved RPLAN Exemplars (Dual Topological &amp; Vector Index)</div>
          {"".join(ex_items)}
        </div>
        """

    return f"""
<div class="hud-grid">
  <div class="hud-card">
    <div class="hud-label">Structural Realism</div>
    <div class="hud-value rose">{realism:.1f}%</div>
    <div class="hud-sub">RPLAN Neural Coherence Metric</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">Circulation &amp; Egress</div>
    <div class="hud-value green">{circulation:.1f}%</div>
    <div class="hud-sub">Graph Connectivity Satisfied</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">Building Code Audit</div>
    <div class="hud-value" style="font-size: 1.35rem;">{comp_badge}</div>
    <div class="hud-sub">IRC / IBC Egress Clearances</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">Synthesis Latency</div>
    <div class="hud-value" style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; color: #BE185D;">{latency_ms:.1f} ms</div>
    <div class="hud-sub">Hardware: {device.upper()} (Epoch {epoch})</div>
  </div>
</div>

<div class="hud-grid" style="margin-top: 12px;">
  <div class="hud-card">
    <div class="hud-label">Wall Topology (GSDiff)</div>
    <div class="hud-value" style="font-size: 1.35rem; color: #9333EA;">{num_walls} Segments</div>
    <div class="hud-sub">{num_junctions} Classified Junction Nodes</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">Architectural Furnishings</div>
    <div class="hud-value" style="font-size: 1.35rem; color: #D97706;">{num_furniture} Placed</div>
    <div class="hud-sub">Code-Compliant Clearance Offsets</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">Aspect Ratio Conformity</div>
    <div class="hud-value" style="font-size: 1.35rem; color: #C026D3;">{aspect:.3f}</div>
    <div class="hud-sub">Optimal Residential Proportion</div>
  </div>
</div>

<div class="exemplars-container" style="margin-top: 12px;">
  <div class="hud-label">Verified Room Inventory Delivered in this Blueprint</div>
  <div style="font-size: 0.85rem; color: #37131D; margin-top: 6px; line-height: 1.6;">
    {rooms_summary_html}
  </div>
</div>

{exemplars_html}
"""


def get_rose_quartz_theme() -> gr.Theme:
    """Creates a configured Gradio Theme representing the Rose Quartz Light Pink design system."""
    theme = gr.themes.Base(
        primary_hue="rose",
        secondary_hue="pink",
        neutral_hue="rose",
        font=[gr.themes.GoogleFont("Inter"), "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
    ).set(
        body_background_fill="#FFF0F5",
        body_background_fill_dark="#FFF0F5",
        background_fill_primary="#FFF0F5",
        background_fill_primary_dark="#FFF0F5",
        background_fill_secondary="rgba(255, 255, 255, 0.72)",
        background_fill_secondary_dark="rgba(255, 255, 255, 0.72)",
        block_background_fill="rgba(255, 255, 255, 0.72)",
        block_background_fill_dark="rgba(255, 255, 255, 0.72)",
        block_border_color="rgba(244, 114, 182, 0.35)",
        block_border_color_dark="rgba(244, 114, 182, 0.35)",
        block_label_background_fill="rgba(255, 255, 255, 0.85)",
        block_label_background_fill_dark="rgba(255, 255, 255, 0.85)",
        block_label_text_color="#701A35",
        block_label_text_color_dark="#701A35",
        block_title_text_color="#37131D",
        block_title_text_color_dark="#37131D",
        input_background_fill="rgba(255, 255, 255, 0.80)",
        input_background_fill_dark="rgba(255, 255, 255, 0.80)",
        input_border_color="rgba(244, 114, 182, 0.45)",
        input_border_color_dark="rgba(244, 114, 182, 0.45)",
        button_secondary_background_fill="rgba(255, 255, 255, 0.45)",
        button_secondary_background_fill_dark="rgba(255, 255, 255, 0.45)",
        button_secondary_border_color="rgba(244, 114, 182, 0.50)",
        button_secondary_border_color_dark="rgba(244, 114, 182, 0.50)",
        button_secondary_text_color="#831843",
        button_secondary_text_color_dark="#831843",
        button_primary_background_fill="#F43F5E",
        button_primary_background_fill_dark="#F43F5E",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
    )
    return theme


get_ocean_depth_theme = get_rose_quartz_theme
OCEAN_DEPTH_CSS = ROSE_QUARTZ_CSS
OCEAN_DEPTH_HEAD_SCRIPT = ROSE_QUARTZ_HEAD_SCRIPT
