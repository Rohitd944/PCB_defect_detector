import { useRef, useState } from "react";
import { FiUploadCloud, FiCamera, FiX, FiImage } from "react-icons/fi";
import "./UploadZone.css";

function UploadZone({ file, onFileChange, onRemove, onOpenCamera }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFileChange(dropped);
  };

  const handleChange = (e) => {
    const selected = e.target.files[0];
    if (selected) onFileChange(selected);
  };

  const openPicker = (e) => {
    e?.stopPropagation();
    inputRef.current?.click();
  };

  const openCamera = (e) => {
    e.stopPropagation();
    onOpenCamera?.();
  };

  if (file) {
    return (
      <div className="upload-zone upload-zone--selected animate-fade-in">
        <div className="upload-zone__file">
          <div className="upload-zone__file-thumb">
            <FiImage size={22} />
          </div>
          <div className="upload-zone__file-info">
            <span className="upload-zone__file-name">{file.name}</span>
            <span className="upload-zone__file-size">
              {(file.size / 1024).toFixed(1)} KB · Ready for inspection
            </span>
          </div>
          <button
            className="upload-zone__remove"
            onClick={onRemove}
            aria-label="Remove file"
          >
            <FiX size={18} />
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          onChange={handleChange}
          style={{ display: "none" }}
          aria-hidden="true"
        />
      </div>
    );
  }

  return (
    <div
      className={`upload-zone ${dragging ? "upload-zone--drag" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={openPicker}
      role="button"
      tabIndex={0}
      aria-label="Upload PCB image"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openPicker();
        }
      }}
    >
      <div className="upload-zone__actions">
        <button type="button" className="upload-zone__action upload-zone__action--upload" onClick={openPicker}>
          <FiUploadCloud size={16} /> Upload
        </button>
        <button type="button" className="upload-zone__action upload-zone__action--camera" onClick={openCamera}>
          <FiCamera size={16} /> Camera
        </button>
      </div>

      <div className="upload-zone__content">
        <div className="upload-zone__icon">
          <FiUploadCloud size={38} />
        </div>
        <div className="upload-zone__text">
          <span className="upload-zone__primary">
            {dragging ? "Release to upload" : "Drop PCB image here"}
          </span>
          <span className="upload-zone__secondary">or click to browse</span>
        </div>
        <div className="upload-zone__formats">
          <span>JPG</span><span>JPEG</span><span>PNG</span>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        onChange={handleChange}
        style={{ display: "none" }}
        aria-hidden="true"
      />
    </div>
  );
}

export default UploadZone;
