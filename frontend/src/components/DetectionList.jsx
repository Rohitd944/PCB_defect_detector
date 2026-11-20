import "./DetectionList.css";

const COLORS = [
  "#2EEE92", "#55EE9E", "#F5B544", "#FA1F21",
  "#7DD3FC", "#C4B5FD", "#FCA5A5", "#86EFAC",
];

function DetectionList({ detections, selectedIndex, onSelect }) {
  if (!detections || detections.length === 0) {
    return (
      <div className="det-list det-list--empty">
        <p className="det-list__empty-title">No Defects Detected</p>
        <p className="det-list__empty-sub mono">Inspection complete</p>
      </div>
    );
  }

  return (
    <div className="det-list">
      <div className="det-list__header">
        <span className="det-list__count mono">{detections.length}</span>
        <span className="det-list__header-label">Detection{detections.length !== 1 ? "s" : ""}</span>
      </div>
      <div className="det-list__rows">
        {detections.map((det, i) => {
          const pct = (det.confidence * 100).toFixed(1);
          const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
          const w = x2 - x1;
          const h = y2 - y1;
          const color = COLORS[i % COLORS.length];
          const isSelected = selectedIndex === i;

          const confidenceLevel =
            det.confidence >= 0.8 ? "high"
            : det.confidence >= 0.5 ? "mid"
            : "low";

          return (
            <div
              key={i}
              className={`det-row ${isSelected ? "det-row--selected" : ""}`}
              style={{ "--row-color": color }}
              onClick={() => onSelect(isSelected ? null : i)}
              role="button"
              tabIndex={0}
              aria-label={`${det.type} defect, ${pct}% confidence`}
              onKeyDown={(e) => e.key === "Enter" && onSelect(isSelected ? null : i)}
            >
              <div className="det-row__accent" />
              <div className="det-row__body">
                <div className="det-row__top">
                  <span className="det-row__type">
                    {det.type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                  </span>
                  <span className={`det-row__conf mono det-row__conf--${confidenceLevel}`}>
                    {pct}%
                  </span>
                </div>
                <div className="det-row__coords mono">
                  <span>x {x1}</span>
                  <span>y {y1}</span>
                  <span>w {w}</span>
                  <span>h {h}</span>
                </div>
                <div className="det-row__bar">
                  <div
                    className="det-row__bar-fill"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default DetectionList;
