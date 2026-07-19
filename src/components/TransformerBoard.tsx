import {
  FFN_EXPANDED_VECTOR,
  LAYER_COUNT,
  STEPS,
  type Simulation,
  generateVector,
  getAttentionHeads,
  getContextVector,
  getFfnOutputVector,
  getVocabulary,
} from '../simulationData';

type BoardProps = {
  sim: Simulation;
  step: number;
  activeTokenIndex: number;
  soloHeadId: string | null;
  onSoloHead: (headId: string | null) => void;
  temperature: number;
};

const TOKEN_Y = 430;
const TOKEN_WIDTH = 86;
const TOKEN_HEIGHT = 38;
const VECTOR_Y = 300;

const makeTokenX = (tokenCount: number) => {
  const slots = tokenCount + 1;
  const gap = slots > 6 ? 100 : 118;
  const startX = 410 - (slots * gap) / 2 + gap / 2;
  return (index: number) => startX + index * gap;
};

const displayToken = (text: string) => (text.length > 9 ? `${text.slice(0, 8)}…` : text);
const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

function valueColor(value: number) {
  const magnitude = Math.min(1, Math.max(0.2, Math.abs(value)));
  const warm = [255, 180, 84];
  const cold = [111, 168, 220];
  const base = [10, 24, 44];
  const target = value >= 0 ? warm : cold;
  const channel = (index: number) => Math.round(base[index] + (target[index] - base[index]) * magnitude);

  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

function HeatStrip({ x, y, values, cellWidth = 9, cellHeight = 24 }: { x: number; y: number; values: number[]; cellWidth?: number; cellHeight?: number }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect x={-6} y={-5} width={values.length * cellWidth + 12} height={cellHeight + 10} rx={2} className="board-vector-shell" />
      {values.map((value, index) => (
        <rect
          className="board-vector-cell"
          fill={valueColor(value)}
          height={cellHeight}
          key={`${value}-${index}`}
          rx={1}
          width={cellWidth - 2}
          x={index * cellWidth}
          y={0}
        />
      ))}
    </g>
  );
}

function SheetChrome() {
  return (
    <g>
      <rect className="board-sheet" height="540" width="820" x="0" y="0" />
      <rect fill="url(#gridMinor)" height="516" width="796" x="12" y="12" />
      <rect fill="url(#gridMajor)" height="516" width="796" x="12" y="12" />
      <rect className="board-frame" height="516" width="796" x="12" y="12" />
    </g>
  );
}

function NotesBlock({ step }: { step: number }) {
  return (
    <g className="board-notes" transform="translate(26 30)">
      <text className="board-notes-label" x={0} y={0}>NOTES</text>
      <text className="board-notes-formula" x={0} y={20}>{STEPS[step].note}</text>
    </g>
  );
}

function TitleBlock({ sim, step, activeTokenIndex }: { sim: Simulation; step: number; activeTokenIndex: number }) {
  const stage = STEPS[step];
  const resolved = step === STEPS.length - 1;
  const figNumber = String(step + 1).padStart(2, '0');

  return (
    <g className="board-titleblock" transform="translate(602 12)">
      <rect className="tb-frame" height={78} width={206} x={0} y={0} />
      <line className="tb-rule" x1={0} x2={206} y1={26} y2={26} />
      <line className="tb-rule" x1={0} x2={206} y1={52} y2={52} />
      <text className="tb-fig" x={10} y={17}>FIG. {figNumber}/10</text>
      <text className="tb-stage" textAnchor="end" x={196} y={17}>{stage.shortTitle.toUpperCase()}</text>
      <text className="tb-meta" x={10} y={43}>QUERY “{displayToken(sim.tokens[activeTokenIndex])}”</text>
      <text className={resolved ? 'tb-meta tb-predict resolved' : 'tb-meta tb-predict'} x={10} y={69}>
        PREDICT {resolved ? `“${sim.nextToken}”` : '?'}
      </text>
      <text className="tb-drawnby" textAnchor="end" x={196} y={69}>P. ACOSTA</text>

      <g className="board-stack-note" transform="translate(0 96)">
        {[2, 1, 0].map((offset) => (
          <rect height={9} key={offset} rx={1} width={26} x={offset * 4} y={-offset * 4} />
        ))}
        <text x={38} y={2}>this block repeats ×{LAYER_COUNT}</text>
      </g>
    </g>
  );
}

function SpecimenNote({ sim, step }: { sim: Simulation; step: number }) {
  if (step !== 0) return null;

  const text = `“${sim.sentence} ___”`;
  const fontSize = Math.max(20, Math.min(38, Math.round(680 / (text.length * 0.52))));

  return (
    <g className="board-specimen stage-group">
      <text className="board-specimen-title" fontSize={fontSize} x={410} y={218}>{text}</text>
      <text className="board-specimen-sub" x={410} y={252}>one forward pass, drawn in ten figures</text>
      <line className="board-specimen-leader" x1={410} x2={410} y1={274} y2={TOKEN_Y - 36} />
    </g>
  );
}

function TokenRow({ sim, step, activeTokenIndex }: { sim: Simulation; step: number; activeTokenIndex: number }) {
  const tokenX = makeTokenX(sim.tokens.length);
  const isAttentionStep = step >= 4 && step <= 6;
  const resolved = step === STEPS.length - 1;
  const ghostX = tokenX(sim.tokens.length);

  return (
    <g>
      {sim.tokenData.map((token, index) => {
        const isActive = index === activeTokenIndex;
        const isMasked = isAttentionStep && index > activeTokenIndex;
        const x = tokenX(index);

        return (
          <g className={isMasked ? 'board-token masked' : isActive ? 'board-token active' : 'board-token'} key={`${token.text}-${index}`}>
            {isActive && isAttentionStep && (
              <circle className="board-token-pulse" cx={x} cy={TOKEN_Y + TOKEN_HEIGHT / 2} r={TOKEN_HEIGHT / 2 + 8} />
            )}
            <rect height={TOKEN_HEIGHT} rx={3} width={TOKEN_WIDTH} x={x - TOKEN_WIDTH / 2} y={TOKEN_Y} />
            <text x={x} y={TOKEN_Y + 24}>{displayToken(token.text)}</text>
            {isActive && <text className="board-token-note" x={x} y={TOKEN_Y + 58}>selected query</text>}
            {isMasked && <text className="board-token-note muted" x={x} y={TOKEN_Y + 58}>masked</text>}
          </g>
        );
      })}

      <g className={resolved ? 'board-ghost resolved' : 'board-ghost'} key={resolved ? 'ghost-resolved' : 'ghost-open'}>
        <rect height={TOKEN_HEIGHT} rx={3} width={TOKEN_WIDTH} x={ghostX - TOKEN_WIDTH / 2} y={TOKEN_Y} />
        <text x={ghostX} y={TOKEN_Y + 24}>{resolved ? displayToken(sim.nextToken) : '?'}</text>
        <text className="board-token-note muted" x={ghostX} y={TOKEN_Y + 58}>{resolved ? 'predicted' : 'next token'}</text>
      </g>
    </g>
  );
}

function EmbeddingStage({ sim, step }: { sim: Simulation; step: number }) {
  if (step < 1 || step > 3) return null;

  const tokenX = makeTokenX(sim.tokens.length);

  return (
    <g>
      {sim.tokenData.map((token, index) => {
        const values = step === 2 ? token.combined : token.embedding;
        const x = tokenX(index) - 36;

        return (
          <g key={`${token.text}-${index}`}>
            <line className="board-flow-line subtle" x1={tokenX(index)} x2={tokenX(index)} y1={TOKEN_Y - 4} y2={VECTOR_Y + 40} />
            <HeatStrip values={values} x={x} y={VECTOR_Y} />
            {step === 2 && (
              <path className="board-position-wave" d={`M ${x - 4} ${VECTOR_Y - 20} c 14 -20, 28 20, 42 0 s 28 -20, 42 0`} />
            )}
          </g>
        );
      })}
    </g>
  );
}

function QkvStage({ sim, step }: { sim: Simulation; step: number }) {
  if (step !== 3) return null;

  const tokenX = makeTokenX(sim.tokens.length);
  const labelX = Math.max(40, tokenX(0) - 52);

  return (
    <g>
      {sim.tokenData.map((token, index) => {
        const x = tokenX(index) - 36;

        return (
          <g key={`${token.text}-${index}`}>
            <line className="board-flow-line subtle" x1={tokenX(index)} x2={tokenX(index)} y1={TOKEN_Y - 4} y2={222} />
            <HeatStrip values={token.qkv.query} x={x} y={196} />
            <HeatStrip values={token.qkv.key} x={x} y={248} />
            <HeatStrip values={token.qkv.value} x={x} y={300} />
          </g>
        );
      })}
      <text className="board-lane-label query" x={labelX} y={214}>Query</text>
      <text className="board-lane-label key" x={labelX} y={266}>Key</text>
      <text className="board-lane-label value" x={labelX} y={318}>Value</text>
    </g>
  );
}

function AttentionStage({ sim, step, activeTokenIndex, soloHeadId, onSoloHead }: { sim: Simulation; step: number; activeTokenIndex: number; soloHeadId: string | null; onSoloHead: (headId: string | null) => void }) {
  if (step < 4 || step > 6) return null;

  const tokenX = makeTokenX(sim.tokens.length);
  const heads = getAttentionHeads(sim, activeTokenIndex);
  const activeX = tokenX(activeTokenIndex);
  const allowed = sim.tokenData.slice(0, activeTokenIndex + 1);
  const isScoreStep = step === 4;
  const isSoftmaxStep = step === 5;
  const isValueStep = step === 6;

  return (
    <g key={step} className="stage-group attention-stage">
      {!isValueStep && (
        <g transform="translate(26 84)">
          {heads.map((head, index) => (
            <g
              className={soloHeadId && soloHeadId !== head.id ? 'board-legend-item dimmed' : 'board-legend-item'}
              key={head.id}
              onClick={() => onSoloHead(soloHeadId === head.id ? null : head.id)}
              transform={`translate(0 ${index * 22})`}
            >
              <rect fill="transparent" height={22} width={200} x={-4} y={-14} />
              <rect fill={head.color} height={10} rx={1} width={16} x={0} y={-8} />
              <text className="board-head-label" x={24} y={1}>{head.label}: {head.description}</text>
            </g>
          ))}
          <text className="board-legend-hint" x={0} y={heads.length * 22 + 8}>click a head to isolate it</text>
        </g>
      )}

      {heads.map((head, headIndex) => {
        const controlY = 105 + headIndex * 18;
        const isDimmed = soloHeadId !== null && soloHeadId !== head.id;

        return (
          <g className={isDimmed ? 'head-dim' : undefined} key={head.id}>
            {head.attention.map((item) => {
              const targetX = tokenX(item.tokenIndex);
              const width = isScoreStep ? Math.max(1.5, Math.abs(item.rawScore) * 1.8) : Math.max(1.5, item.weight * 9);
              const topY = 175 + headIndex * 34;
              const pathD = `M ${activeX} ${TOKEN_Y - 12} C ${activeX} ${controlY}, ${targetX} ${controlY}, ${targetX} ${TOKEN_Y - 12}`;

              return (
                <g key={`${head.id}-${item.tokenIndex}`}>
                  <path
                    className={`board-attention-path ${isScoreStep ? 'scoring' : ''} ${isValueStep ? 'value-flow' : ''}`}
                    d={pathD}
                    stroke={head.color}
                    strokeDasharray={isScoreStep ? '7 7' : undefined}
                    strokeWidth={width}
                  />
                  {(isScoreStep || isSoftmaxStep) && (
                    <text className="board-attention-label" fill={head.color} x={targetX} y={topY}>
                      {isScoreStep ? item.rawScore.toFixed(2) : formatPercent(item.weight)}
                    </text>
                  )}
                  {isValueStep && (
                    <circle className="board-value-particle" fill={head.color} r={Math.max(3, item.weight * 6)}>
                      <animateMotion dur={`${1.2 - item.weight * 0.6}s`} repeatCount="indefinite" path={pathD} />
                    </circle>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}

      {isValueStep && (
        <g transform="translate(614 150)">
          {heads.map((head, index) => (
            <g
              className={soloHeadId && soloHeadId !== head.id ? 'board-legend-item head-dim' : 'board-legend-item'}
              key={head.id}
              onClick={() => onSoloHead(soloHeadId === head.id ? null : head.id)}
              transform={`translate(0 ${index * 52})`}
            >
              <text className="board-head-output" fill={head.color} x={0} y={17}>{head.label}</text>
              <HeatStrip values={head.attention.map((item) => item.weight * 2 - 1)} x={68} y={0} />
            </g>
          ))}
          <text className="board-merge-label" x={0} y={186}>concat + project</text>
        </g>
      )}

      {activeTokenIndex < sim.tokens.length - 1 && (
        <text className="board-mask-note" x={tokenX(activeTokenIndex + 1)} y={TOKEN_Y - 30}>
          future tokens are masked
        </text>
      )}

      {allowed.length === 1 && (
        <text className="board-mask-note" x={activeX - 56} y={TOKEN_Y - 72}>
          first token can only attend to itself
        </text>
      )}
    </g>
  );
}

function FlowLines({ sim, step, activeTokenIndex }: { sim: Simulation; step: number; activeTokenIndex: number }) {
  if (step === 0) return null;

  const tokenX = makeTokenX(sim.tokens.length);
  const activeX = tokenX(activeTokenIndex);
  const stageCenters: Record<number, { x: number; y: number }> = {
    1: { x: activeX, y: VECTOR_Y + 12 },
    2: { x: activeX, y: VECTOR_Y + 12 },
    3: { x: activeX, y: 248 },
    4: { x: activeX, y: 175 },
    5: { x: activeX, y: 175 },
    6: { x: activeX, y: 175 },
    7: { x: 240, y: 180 },
    8: { x: 300, y: 160 },
    9: { x: 300, y: 160 },
  };

  return (
    <g className="board-flow-lines">
      {Object.entries(stageCenters).map(([stageIndex, center]) => {
        const isCurrent = Number(stageIndex) === step;
        const isPast = Number(stageIndex) < step;
        const startX = activeX;
        const startY = TOKEN_Y - 12;
        const midY = (startY + center.y) / 2;

        return (
          <path
            key={stageIndex}
            className={`board-flow-line animated ${isCurrent ? 'active' : ''} ${isPast ? 'past' : ''}`}
            d={`M ${startX} ${startY} C ${startX} ${midY}, ${center.x} ${midY}, ${center.x} ${center.y}`}
          />
        );
      })}
    </g>
  );
}

function FfnStage({ sim, step, activeTokenIndex }: { sim: Simulation; step: number; activeTokenIndex: number }) {
  if (step !== 7) return null;

  return (
    <g transform="translate(124 178)">
      <HeatStrip values={getContextVector(sim, activeTokenIndex)} x={0} y={0} />
      <text className="board-stage-caption" x={36} y={-18}>context</text>
      <path className="board-flow-line" d="M 100 14 L 190 14" />
      <HeatStrip cellWidth={7} values={FFN_EXPANDED_VECTOR} x={210} y={0} />
      <text className="board-stage-caption" x={260} y={-18}>expanded hidden layer</text>
      <path className="board-flow-line" d="M 342 14 L 432 14" />
      <HeatStrip values={getFfnOutputVector(sim, activeTokenIndex)} x={452} y={0} />
      <text className="board-stage-caption" x={488} y={-18}>output</text>
    </g>
  );
}

function VocabularyStage({ sim, step, temperature }: { sim: Simulation; step: number; temperature: number }) {
  if (step < 8) return null;

  const isOutputStep = step === 9;
  const vocabulary = getVocabulary(sim, isOutputStep ? temperature : 1);

  return (
    <g transform="translate(104 142)">
      <g>
        {[generateVector(601, 18), generateVector(641, 18), generateVector(681, 18)].map((row, index) => (
          <HeatStrip cellWidth={7} key={index} values={row} x={0} y={index * 42} />
        ))}
      </g>
      <path className="board-flow-line" d="M 170 52 L 260 52" />
      <text className="board-stage-caption" x={24} y={-20}>vocabulary matrix</text>
      <g transform="translate(294 -8)">
        {vocabulary.map((item, index) => {
          const width = isOutputStep ? Math.max(2, item.probability * 270) : Math.max(28, item.logit * 62);
          const isWinner = item.token === sim.nextToken;

          return (
            <g className={isWinner ? 'board-vocab-row winner' : 'board-vocab-row'} key={item.token} transform={`translate(0 ${index * 42})`}>
              <text x={0} y={20}>{displayToken(item.token)}</text>
              <rect height={18} rx={2} width={width} x={70} y={6} />
              <text className="board-vocab-value" x={width + 84} y={20}>{isOutputStep ? formatPercent(item.probability) : item.logit.toFixed(2)}</text>
            </g>
          );
        })}
      </g>
    </g>
  );
}

export default function TransformerBoard({ sim, step, activeTokenIndex, soloHeadId, onSoloHead, temperature }: BoardProps) {
  return (
    <div className="board-shell" aria-label="Transformer process visualization">
      <svg className="transformer-board" role="img" viewBox="0 0 820 540">
        <defs>
          <pattern height="20" id="gridMinor" patternUnits="userSpaceOnUse" width="20">
            <path className="board-grid-minor" d="M 20 0 L 0 0 0 20" fill="none" />
          </pattern>
          <pattern height="100" id="gridMajor" patternUnits="userSpaceOnUse" width="100">
            <path className="board-grid-major" d="M 100 0 L 0 0 0 100" fill="none" />
          </pattern>
        </defs>
        <SheetChrome />
        <SpecimenNote sim={sim} step={step} />
        <FlowLines activeTokenIndex={activeTokenIndex} sim={sim} step={step} />
        <EmbeddingStage sim={sim} step={step} />
        <QkvStage sim={sim} step={step} />
        <AttentionStage activeTokenIndex={activeTokenIndex} onSoloHead={onSoloHead} sim={sim} soloHeadId={soloHeadId} step={step} />
        <FfnStage activeTokenIndex={activeTokenIndex} sim={sim} step={step} />
        <VocabularyStage sim={sim} step={step} temperature={temperature} />
        <TokenRow activeTokenIndex={activeTokenIndex} sim={sim} step={step} />
        <NotesBlock step={step} />
        <TitleBlock activeTokenIndex={activeTokenIndex} sim={sim} step={step} />
      </svg>
    </div>
  );
}
