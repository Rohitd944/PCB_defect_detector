import { useRef, useState } from "react";
import {
  FiUploadCloud,
  FiCamera,
  FiX,
  FiImage,
  FiCheckCircle,
  FiAlertTriangle,
} from "react-icons/fi";
import "./ImageInputCard.css";

/**
 * Reusable image input card: drag & drop + upload + camera + preview.
 * Used twice (reference PCB and testing PCB) in the hybrid workspace.
 */
function ImageInputCard({
  title,
  subtitle,
  badge,
  tone = "accent",
  file,
  preview,
  onFileChange,
  onRemove,
  onOpenCamera,
  status,
  statusTone = "ok",
  disabled = false,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFileChange(dropped);
  };

  const handleChange = (event) => {
    const selected = event.target.files?.[0];
    if (selected) onFileChange(selected);
    event.target.value = "";
  };

  return (
    <section
      className={`input-card input-card--${tone} ${dragging ? "input-card--drag" : ""} ${preview ? "input-card--filled" : ""}`}
      aria-label={title}
    >
      <header className="input-card__header">
        <div className="input-card__heading">
          <span className={`input-card__badge input-card__badge--${tone}`}>{badge}</span>
          <h3 className="input-card__title">{title}</h3>
          <p className="input-card__subtitle">{subtitle}</p>
        </div>
        {preview && (
          <button
            type="button"
            className="input-card__remove"
            onClick={onRemove}
            aria-label={`Remove ${title}`}
            disabled={disabled}
          >
            <FiX size={16} />
          </button>
        )}
      </header>

      <div
        className={`input-card__body ${dragging ? "input-card__body--drag" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && !preview && inputRef.current?.click()}
        role={!preview ? "button" : undefined}
        tabIndex={!preview ? 0 : undefined}
        onKeyDown={(event) => {
          if (!preview && event.key === "Enter" && !disabled) inputRef.current?.click();
        }}
      >
        {preview ? (
          <img className="input-card__preview" src={preview} alt={`${title} preview`} />
        ) : (
          <div className="input-card__placeholder">
            <div className="input-card__placeholder-icon">
              <FiImage size={26} />
            </div>
            <span className="input-card__placeholder-title">
              {dragging ? "Release to load image" : "Drop image here"}
            </span>
            <span className="input-card__placeholder-sub">
              or click to browse · JPG · PNG
            </span>
          </div>
        )}
        {dragging && <div className="input-card__drop-overlay">DROP IMAGE</div>}
      </div>

      <div className="input-card__actions">
        <button
          type="button"
          className="input-card__btn"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
        >
          <FiUploadCloud size={15} />
          {file ? "Replace Image" : "Upload Image"}
        </button>
        <button
          type="button"
          className="input-card__btn input-card__btn--camera"
          onClick={onOpenCamera}
          disabled={disabled}
        >
          <FiCamera size={15} />
          Open Camera
        </button>
      </div>

      {status && (
        <div className={`input-card__status input-card__status--${statusTone}`}>
          {statusTone === "ok" ? <FiCheckCircle size={13} /> : <FiAlertTriangle size={13} />}
          <span>{status}</span>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        onChange={handleChange}
        style={{ display: "none" }}
        aria-hidden="true"
      />
    </section>
  );
}

export default ImageInputCard;
