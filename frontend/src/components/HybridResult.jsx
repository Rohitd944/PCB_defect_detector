import { useState } from "react";
import {
  FiCheckCircle,
  FiXCircle,
  FiAlertTriangle,
  FiDownload,
  FiRefreshCw,
  FiMaximize2,
  FiX,
  FiCpu,
  FiCrosshair,
} from "react-icons/fi";
import "./HybridResult.css";

const VERDICT_META = {
  PASS: { icon: FiCheckCircle, tone: "pass", label: "PASS" },
  FAIL: { icon: FiXCircle, tone: "fail", label: "FAIL" },
  REVIEW: { icon: FiAlertTriangle, tone: "review", label: "REVIEW" },
};

const VERIFICATION_LABEL = {
  confirmed: { text: "Confirmed", tone: "confirm" },
  rejected: { text: "Rejected", tone: "reject" },
  unverified: { text: "Unverified", tone: "warn" },
  not_compared: { text: "Not compared", tone: "muted" },
};

function pct(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function num(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
}

function HybridResult({ result, onReset }) {
  const [lightbox, setLightbox] = useState(null);

  const verdict = result?.verdict || "REVIEW";
  const meta = VERDICT_META[verdict] || VERDICT_META.REVIEW;
  const VerdictIcon = meta.icon;

  const diff = result?.diff;
  const alignment = result?.alignment;
  const summary = result?.summary || {};
  const detections = result?.detections || [];
  const anomalies = result?.anomalies || [];
  const images = result?.images || {};

  const tiles = [
    { key: "annotated_url", label: "Annotated Testing PCB", url: images.annotated_url },
    { key: "comparison_url", label: "Reference vs Testing", url: images.comparison_url },
    { key: "heatmap_url", label: "Difference Heatmap", url: images.heatmap_url },
  ].filter((tile) => tile.url);

  return (
    <div className="hresult animate-fade-in">
      {/* Verdict banner */}
      <div className={`hresult__banner hresult__banner--${meta.tone}`}>
        <div className="hresult__banner-main">
          <VerdictIcon size={30} className="hresult__banner-icon" />
          <div>
            <div className="hresult__banner-label mono">HYBRID INSPECTION</div>
            <h2 className="hresult__banner-verdict">{meta.label}</h2>
          </div>
        </div>
        <p className="hresult__banner-reason">{result?.verdict_reason}</p>
        <div className="hresult__banner-meta mono">
          <span>{result?.inspection_id}</span>
          <span>{result?.timestamp}</span>
          {result?.confidence_level && <span>CONFIDENCE {result.confidence_level.toUpperCase()}</span>}
        </div>
      </div>

      {/* Metrics */}
      <div className="hresult__metrics">
        <MetricCard
          label="Reference Match (SSIM)"
          value={diff ? num(diff.ssim, 4) : "—"}
          hint="1.000 = identical structure"
          tone={diff && diff.ssim >= 0.995 ? "ok" : diff && diff.ssim >= 0.98 ? "warn" : "bad"}
        />
        <MetricCard
          label="Difference Score"
          value={diff ? `${num(diff.difference_score, 0)}/100` : "—"}
          hint={diff ? `${num(diff.changed_pixel_pct, 2)}% of pixels changed` : "not computed"}
          tone={!diff ? "muted" : diff.difference_score >= 40 ? "bad" : diff.difference_score >= 15 ? "warn" : "ok"}
        />
        <MetricCard
          label="Alignment"
          value={alignment ? alignment.quality.toUpperCase() : "—"}
          hint={
            alignment
              ? `${alignment.inliers} inliers · rot ${num(alignment.rotation_deg, 1)}° · ${num(alignment.scale, 2)}×`
              : "—"
          }
          tone={!alignment ? "muted" : alignment.quality === "good" ? "ok" : alignment.quality === "weak" ? "warn" : "bad"}
        />
        <MetricCard
          label="Verification"
          value={`${summary.confirmed ?? 0} / ${summary.total_detections ?? 0}`}
          hint={`confirmed · ${summary.rejected ?? 0} rejected · ${summary.anomalies ?? 0} anomalies`}
          tone={summary.confirmed > 0 ? "bad" : summary.anomalies > 0 ? "warn" : "ok"}
        />
      </div>

      {/* Images */}
      {tiles.length > 0 && (
        <div className="hresult__images">
          {tiles.map((tile) => (
            <figure className="hresult__figure" key={tile.key}>
              <div className="hresult__figure-head">
                <figcaption className="hresult__figure-label">{tile.label}</figcaption>
                <div className="hresult__figure-actions">
                  <a
                    href={tile.url}
                    download
                    className="hresult__icon-btn"
                    aria-label={`Download ${tile.label}`}
                  >
                    <FiDownload size={14} />
                  </a>
                  <button
                    type="button"
                    className="hresult__icon-btn"
                    onClick={() => setLightbox({ url: tile.url, label: tile.label })}
                    aria-label={`Enlarge ${tile.label}`}
                  >
                    <FiMaximize2 size={14} />
                  </button>
                </div>
              </div>
              <button
                type="button"
                className="hresult__figure-body"
                onClick={() => setLightbox({ url: tile.url, label: tile.label })}
              >
                <img src={tile.url} alt={tile.label} loading="lazy" />
              </button>
            </figure>
          ))}
        </div>
      )}

      {/* Detections */}
      <div className="hresult__panel">
        <div className="hresult__panel-head">
          <span className="label-chip">
            <FiCpu size={11} /> YOLO Detections &amp; Verification
          </span>
          <span className="hresult__panel-count mono">{detections.length}</span>
        </div>

        {detections.length === 0 ? (
          <p className="hresult__empty">
            No YOLO detections at {Math.round((result?.parameters?.confidence ?? 0.25) * 100)}%
            confidence.
          </p>
        ) : (
          <div className="hresult__table-wrap">
            <table className="hresult__table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Defect</th>
                  <th>YOLO</th>
                  <th>Ref. Difference</th>
                  <th>Location</th>
                  <th>Verification</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((det, index) => {
                  const verification = VERIFICATION_LABEL[det.verification] || VERIFICATION_LABEL.not_compared;
                  return (
                    <tr key={index}>
                      <td className="mono">{String(index + 1).padStart(2, "0")}</td>
                      <td className="hresult__cell-type">{det.type}</td>
                      <td className="mono">{pct(det.confidence)}</td>
                      <td className="mono">
                        {det.reference_difference !== null && det.reference_difference !== undefined
                          ? num(det.reference_difference, 1)
                          : "—"}
                        {det.matched_regions ? (
                          <span className="hresult__region-count">
                            {det.matched_regions} region{det.matched_regions > 1 ? "s" : ""}
                          </span>
                        ) : null}
                      </td>
                      <td className="mono">
                        {det.location ? `${det.location.x}% × ${det.location.y}%` : "—"}
                      </td>
                      <td>
                        <span className={`hresult__pill hresult__pill--${verification.tone}`}>
                          {verification.text}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Anomalies */}
      <div className="hresult__panel">
        <div className="hresult__panel-head">
          <span className="label-chip">
            <FiCrosshair size={11} /> Reference Anomalies (no YOLO match)
          </span>
          <span className="hresult__panel-count mono">{anomalies.length}</span>
        </div>

        {anomalies.length === 0 ? (
          <p className="hresult__empty">No unmatched structural differences above tolerance.</p>
        ) : (
          <div className="hresult__table-wrap">
            <table className="hresult__table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Severity</th>
                  <th>Mean Diff</th>
                  <th>Area</th>
                  <th>Bounding Box</th>
                  <th>SSIM</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.map((item, index) => (
                  <tr key={index}>
                    <td className="mono">{String(index + 1).padStart(2, "0")}</td>
                    <td>
                      <span className="hresult__severity">
                        <span
                          className="hresult__severity-bar"
                          style={{ width: `${Math.min(100, item.severity)}%` }}
                        />
                        <span className="mono">{num(item.severity, 0)}</span>
                      </span>
                    </td>
                    <td className="mono">{num(item.mean_diff, 1)}</td>
                    <td className="mono">{item.area} px</td>
                    <td className="mono">[{item.bbox.join(", ")}]</td>
                    <td className="mono">{num(item.ssim, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Alignment detail */}
      {alignment && (
        <div className="hresult__panel">
          <div className="hresult__panel-head">
            <span className="label-chip">Alignment Detail</span>
          </div>
          <div className="hresult__kv">
            <KV label="Quality" value={alignment.quality} />
            <KV label="Good matches" value={alignment.good_matches} />
            <KV label="Inliers" value={`${alignment.inliers} (${num(alignment.inlier_ratio * 100, 1)}%)`} />
            <KV label="Reprojection" value={`${num(alignment.reprojection_error, 2)} px`} />
            <KV label="Rotation" value={`${num(alignment.rotation_deg, 2)}°`} />
            <KV label="Scale" value={`${num(alignment.scale, 3)}×`} />
            <KV label="Overlap" value={pct(alignment.coverage)} />
            <KV
              label="Reference size"
              value={`${alignment.reference_size?.width}×${alignment.reference_size?.height}`}
            />
          </div>
          {alignment.warnings && alignment.warnings.length > 0 && (
            <ul className="hresult__warnings">
              {alignment.warnings.map((warning, index) => (
                <li key={index}>
                  <FiAlertTriangle size={12} /> {warning}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="hresult__final-actions">
        {images.annotated_url && (
          <a href={images.annotated_url} download className="ws-btn ws-btn--ghost">
            <FiDownload size={16} /> Download Annotated
          </a>
        )}
        {images.heatmap_url && (
          <a href={images.heatmap_url} download className="ws-btn ws-btn--ghost">
            <FiDownload size={16} /> Download Heatmap
          </a>
        )}
        <button type="button" className="ws-btn ws-btn--ghost" onClick={onReset}>
          <FiRefreshCw size={16} /> New Comparison
        </button>
      </div>

      {lightbox && (
        <div
          className="hresult__lightbox"
          role="dialog"
          aria-label={lightbox.label}
          onClick={() => setLightbox(null)}
        >
          <button
            type="button"
            className="hresult__lightbox-close"
            onClick={() => setLightbox(null)}
            aria-label="Close preview"
          >
            <FiX size={22} />
          </button>
          <figure className="hresult__lightbox-figure" onClick={(event) => event.stopPropagation()}>
            <img src={lightbox.url} alt={lightbox.label} />
            <figcaption>{lightbox.label}</figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, hint, tone = "muted" }) {
  return (
    <div className={`hresult__metric hresult__metric--${tone}`}>
      <span className="hresult__metric-label">{label}</span>
      <span className="hresult__metric-value mono">{value}</span>
      <span className="hresult__metric-hint">{hint}</span>
    </div>
  );
}

function KV({ label, value }) {
  return (
    <div className="hresult__kv-item">
      <span className="hresult__kv-label">{label}</span>
      <span className="hresult__kv-value mono">{value}</span>
    </div>
  );
}

export default HybridResult;
