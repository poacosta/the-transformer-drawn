import { useEffect, useMemo, useState } from 'react';
import { Play, Pause, ChevronLeft, ChevronRight, Download, PenLine } from 'lucide-react';
import TransformerBoard from './components/TransformerBoard';
import { DEFAULT_SENTENCE, MAX_TOKENS, STEPS, buildSimulation, getAttentionHeads, getReadoutRows } from './simulationData';
import { exportSheet } from './exportSheet';
import './index.css';

function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sentence, setSentence] = useState(DEFAULT_SENTENCE);
  const [draft, setDraft] = useState(DEFAULT_SENTENCE);
  const [activeTokenIndex, setActiveTokenIndex] = useState(2);
  const [soloHeadId, setSoloHeadId] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(1);

  const sim = useMemo(() => buildSimulation(sentence), [sentence]);
  const activeStep = STEPS[currentStep];
  const readoutRows = getReadoutRows(sim, currentStep, activeTokenIndex, temperature);
  const heads = getAttentionHeads(sim, activeTokenIndex);
  const isLastStep = currentStep === STEPS.length - 1;
  const isAttentionStep = currentStep >= 4 && currentStep <= 6;

  const nextStep = () => {
    setCurrentStep((prev) => {
      const next = Math.min(prev + 1, STEPS.length - 1);
      if (next === STEPS.length - 1) {
        setIsPlaying(false);
      }
      return next;
    });
  };

  const prevStep = () => {
    setIsPlaying(false);
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  };

  const togglePlay = () => {
    if (isLastStep && !isPlaying) {
      setCurrentStep(0);
    }
    setIsPlaying((playing) => !playing);
  };

  const commitSentence = () => {
    const tokens = draft.trim().split(/\s+/).filter(Boolean).slice(0, MAX_TOKENS);
    if (tokens.length === 0) return;

    const nextSentence = tokens.join(' ');
    setDraft(nextSentence);
    if (nextSentence === sentence) return;

    setIsPlaying(false);
    setSentence(nextSentence);
    setActiveTokenIndex(Math.min(2, tokens.length - 1));
    setCurrentStep(0);
    setSoloHeadId(null);
  };

  const toggleSoloHead = (headId: string | null) => {
    setSoloHeadId((current) => (current === headId ? null : headId));
  };

  useEffect(() => {
    if (!isPlaying) return undefined;

    const timer = window.setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= STEPS.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 2600);

    return () => window.clearInterval(timer);
  }, [isPlaying]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return;
      }

      if (event.key === 'ArrowRight') {
        event.preventDefault();
        nextStep();
      }

      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        prevStep();
      }

      if (event.key === ' ' && target?.tagName !== 'BUTTON') {
        event.preventDefault();
        togglePlay();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  return (
    <div className="app-frame">
      <header className="masthead">
        <div className="masthead-brand">
          <span className="masthead-eyebrow">Working drawing · GPT-style decoder block</span>
          <h1>The Transformer, Drawn</h1>
        </div>

        <div className="masthead-controls">
          <form
            className="specimen-form"
            onSubmit={(event) => {
              event.preventDefault();
              commitSentence();
              (event.currentTarget.querySelector('input') as HTMLInputElement | null)?.blur();
            }}
          >
            <label className="picker-label" htmlFor="specimen-input">Specimen · max {MAX_TOKENS} words</label>
            <div className="specimen-row">
              <input
                id="specimen-input"
                className="specimen-input"
                onChange={(event) => setDraft(event.target.value)}
                placeholder={DEFAULT_SENTENCE}
                spellCheck={false}
                type="text"
                value={draft}
              />
              <button className="specimen-submit" type="submit" aria-label="Redraw with this sentence">
                <PenLine size={14} /> Redraw
              </button>
            </div>
          </form>

          <div className="query-picker-group">
            <span className="picker-label" id="token-picker-label">Query token</span>
            <div aria-labelledby="token-picker-label" className="token-picker" role="group">
              {sim.tokens.map((token, index) => (
                <button
                  className={`token-picker-button ${index === activeTokenIndex ? 'active' : ''}`}
                  key={`${token}-${index}`}
                  onClick={() => {
                    setIsPlaying(false);
                    setActiveTokenIndex(index);
                  }}
                  aria-pressed={index === activeTokenIndex}
                >
                  {token}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="workspace">
        <section className="drawing-area">
          <TransformerBoard
            activeTokenIndex={activeTokenIndex}
            onSoloHead={toggleSoloHead}
            sim={sim}
            soloHeadId={soloHeadId}
            step={currentStep}
            temperature={temperature}
          />
        </section>

        <aside className="margin-notes">
          <section className="note-card animate-fade-in" key={`${activeStep.id}-${sentence}`}>
            <div className="note-kicker">Fig. {String(currentStep + 1).padStart(2, '0')} · {activeStep.shortTitle}</div>
            <h2>{activeStep.title}</h2>
            <p>{activeStep.description}</p>
          </section>

          <section className="note-card readout" aria-label={`${activeStep.title} readout`}>
            <div className="note-kicker">{activeStep.readoutTitle}</div>
            <div className="readout-rows">
              {readoutRows.map((row) => (
                <div className="readout-row" key={`${row.label}-${row.value}`}>
                  <span>{row.label}</span>
                  <strong>{row.value}</strong>
                </div>
              ))}
            </div>

            {isLastStep && (
              <div className="temp-control">
                <label htmlFor="temperature-slider">
                  Sampling temperature · T = {temperature.toFixed(2)}
                </label>
                <input
                  id="temperature-slider"
                  max={3}
                  min={0.2}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                  step={0.05}
                  type="range"
                  value={temperature}
                />
                <div className="temp-scale">
                  <span>greedy</span>
                  <span>random</span>
                </div>
              </div>
            )}
          </section>

          <section className="note-card legend" aria-label="Color code">
            <div className="note-kicker">Color code</div>
            <ul className="legend-list">
              <li><span className="legend-swatch positive" /> positive feature</li>
              <li><span className="legend-swatch negative" /> negative feature</li>
              {heads.map((head) => (
                <li key={head.id}>
                  <button
                    className={`legend-head-button ${soloHeadId && soloHeadId !== head.id ? 'dimmed' : ''} ${soloHeadId === head.id ? 'solo' : ''}`}
                    onClick={() => toggleSoloHead(head.id)}
                    aria-pressed={soloHeadId === head.id}
                  >
                    <span className="legend-swatch" style={{ background: head.color }} /> {head.label.toLowerCase()} · {head.description}
                  </button>
                </li>
              ))}
            </ul>
            {isAttentionStep && <p className="legend-hint">click a head to isolate its attention</p>}
          </section>
        </aside>
      </main>

      <footer className="transport">
        <nav className="timeline" aria-label="Transformer lesson steps">
          {STEPS.map((step, idx) => (
            <button
              className={`timeline-step ${idx === currentStep ? 'active' : ''} ${idx < currentStep ? 'complete' : ''}`}
              key={step.id}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep(idx);
              }}
              aria-current={idx === currentStep ? 'step' : undefined}
            >
              <span>{String(idx + 1).padStart(2, '0')}</span>
              {step.shortTitle}
            </button>
          ))}
        </nav>

        <div className="controls">
          <button onClick={prevStep} disabled={currentStep === 0} aria-label="Previous step">
            <ChevronLeft size={16} /> Prev
          </button>

          <button className="primary" onClick={togglePlay} aria-label={isPlaying ? 'Pause lesson' : 'Play lesson'}>
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? 'Pause' : isLastStep ? 'Replay' : 'Play'}
          </button>

          <button onClick={nextStep} disabled={isLastStep} aria-label="Next step">
            Next <ChevronRight size={16} />
          </button>

          <button onClick={() => exportSheet(currentStep + 1)} aria-label="Download this sheet as an SVG file">
            <Download size={16} /> Sheet
          </button>
        </div>

        <p className="colophon">
          Created by <a href="https://github.com/poacosta" rel="noreferrer" target="_blank">Pedro Acosta</a> · Released under the MIT License
        </p>
      </footer>
    </div>
  );
}

export default App;
