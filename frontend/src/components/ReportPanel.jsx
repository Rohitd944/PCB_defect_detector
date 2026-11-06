import { useState } from "react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import { FiFileText, FiDownload, FiLoader } from "react-icons/fi";
import "./ReportPanel.css";

function ReportPanel({ analysis, detections, imageUrl }) {
  const [state, setState] = useState("idle"); // idle | generating | done | error

  if (!analysis) return null;

  const loadImage = (url) =>
    new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });

  const downloadPDF = async () => {
    setState("generating");
    try {
      const doc = new jsPDF();
      let y = 20;

      const checkBreak = (space = 20) => {
        if (y + space > doc.internal.pageSize.getHeight() - 20) {
          doc.addPage();
          y = 20;
        }
      };

      // Header
      doc.setFont("helvetica", "bold");
      doc.setFontSize(22);
      doc.text("PCB INSPECTION REPORT", 20, y);
      y += 10;
      doc.setDrawColor(180, 180, 180);
      doc.setLineWidth(0.5);
      doc.line(20, y, 190, y);
      y += 10;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      doc.text(`Report ID  : ${analysis.inspection_id}`, 20, y); y += 7;
      doc.text(`Generated  : ${analysis.timestamp}`,     20, y); y += 7;
      doc.text(`Model      : YOLOv8`,                    20, y); y += 7;
      doc.text(`File       : ${analysis.filename}`,      20, y); y += 14;

      // Summary
      checkBreak(40);
      doc.setFont("helvetica", "bold"); doc.setFontSize(16);
      doc.text("INSPECTION SUMMARY", 20, y); y += 10;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      doc.text(`Total Defects        : ${analysis.total_defects}`,        20, y); y += 7;
      doc.text(`Highest Confidence   : ${analysis.highest_confidence}%`,  20, y); y += 7;
      doc.text(`Average Confidence   : ${analysis.average_confidence}%`,  20, y); y += 7;
      doc.text(`Inspection Status    : ${analysis.inspection_status}`,    20, y); y += 14;

      // Recommendation
      checkBreak(40);
      doc.setFont("helvetica", "bold"); doc.setFontSize(16);
      doc.text("AI RECOMMENDATION", 20, y); y += 10;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      const recLines = doc.splitTextToSize(analysis.recommendation, 170);
      doc.text(recLines, 20, y);
      y += recLines.length * 7 + 12;

      // Observations
      checkBreak(40);
      doc.setFont("helvetica", "bold"); doc.setFontSize(16);
      doc.text("INSPECTION OBSERVATIONS", 20, y); y += 10;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      analysis.observations.forEach((obs) => {
        checkBreak(10);
        doc.text(`• ${obs}`, 25, y); y += 7;
      });
      y += 10;

      // Detection table
      checkBreak(60);
      doc.setFont("helvetica", "bold"); doc.setFontSize(16);
      doc.text("DETECTION TABLE", 20, y); y += 10;
      autoTable(doc, {
        startY: y,
        head: [["#", "Defect Type", "Confidence", "Bounding Box"]],
        body: detections && detections.length > 0
          ? detections.map((d, i) => [
              i + 1,
              d.type,
              `${(d.confidence * 100).toFixed(2)}%`,
              d.bbox ? `[${d.bbox.join(", ")}]` : "—",
            ])
          : [["—", "No manufacturing defects detected", "—", "—"]],
        theme: "grid",
        headStyles: { fillColor: [23, 28, 28], textColor: 240, halign: "center" },
        bodyStyles: { halign: "center", fontSize: 10 },
      });
      y = doc.lastAutoTable.finalY + 15;

      // Annotated image
      checkBreak(120);
      doc.setFont("helvetica", "bold"); doc.setFontSize(16);
      doc.text("ANNOTATED PCB RESULT", 20, y); y += 12;
      try {
        const img = await loadImage(imageUrl);
        const canvas = document.createElement("canvas");
        canvas.width  = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        const imgData = canvas.toDataURL("image/jpeg");
        const maxW = 170, maxH = 110;
        const ratio = Math.min(maxW / img.width, maxH / img.height);
        const iw = img.width * ratio;
        const ih = img.height * ratio;
        checkBreak(ih + 20);
        doc.addImage(imgData, "JPEG", (210 - iw) / 2, y, iw, ih);
        y += ih + 10;
        doc.setFont("helvetica", "italic"); doc.setFontSize(10);
        doc.text("Figure 1 — YOLOv8 PCB Defect Detection Result", 105, y, { align: "center" });
        y += 14;
      } catch {
        doc.setTextColor(200, 0, 0);
        doc.text("Unable to load annotated PCB image.", 20, y);
        doc.setTextColor(0, 0, 0);
        y += 10;
      }

      // Footer
      checkBreak(30);
      doc.setDrawColor(180); doc.line(20, y, 190, y); y += 10;
      doc.setFont("helvetica", "normal"); doc.setFontSize(10);
      doc.text("AI Powered PCB Defect Detection System — CircuitSentinel", 20, y); y += 6;
      doc.text(`Report ID: ${analysis.inspection_id}`, 20, y); y += 6;
      doc.text("© 2026 CircuitSentinel", 20, y);

      doc.save(`PCB_Report_${analysis.inspection_id}.pdf`);
      setState("done");
    } catch {
      setState("error");
    }
  };

  return (
    <div className="report-panel animate-fade-in">
      <div className="report-panel__header">
        <FiFileText size={16} className="report-panel__icon" />
        <span className="report-panel__title">Inspection Report</span>
        <span className={`report-panel__badge mono report-panel__badge--${analysis.inspection_status.toLowerCase().replace(" ", "-")}`}>
          {analysis.inspection_status}
        </span>
      </div>

      <div className="report-panel__body">
        <div className="report-panel__recommendation">
          <span className="report-panel__rec-label mono">AI RECOMMENDATION</span>
          <p className="report-panel__rec-text">{analysis.recommendation}</p>
        </div>

        <div className="report-panel__observations">
          <span className="report-panel__obs-label mono">OBSERVATIONS</span>
          <ul className="report-panel__obs-list">
            {analysis.observations.map((obs, i) => (
              <li key={i} className="report-panel__obs-item">{obs}</li>
            ))}
          </ul>
        </div>

        {analysis.distribution && Object.keys(analysis.distribution).length > 0 && (
          <div className="report-panel__distribution">
            <span className="report-panel__dist-label mono">DEFECT DISTRIBUTION</span>
            <div className="report-panel__dist-rows">
              {Object.entries(analysis.distribution).map(([type, count]) => (
                <div key={type} className="report-panel__dist-row">
                  <span className="report-panel__dist-type">{type.replace(/_/g, " ")}</span>
                  <span className="report-panel__dist-count mono">{count.toString().padStart(2, "0")}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="report-panel__footer">
        <button
          className={`report-btn report-btn--${state}`}
          onClick={downloadPDF}
          disabled={state === "generating"}
        >
          {state === "generating" ? (
            <><FiLoader size={15} className="animate-spin" /> Generating Report...</>
          ) : state === "done" ? (
            <><FiDownload size={15} /> Download Again</>
          ) : (
            <><FiDownload size={15} /> Download Inspection Report</>
          )}
        </button>
        {state === "error" && (
          <p className="report-btn__error">Report generation failed. Try again.</p>
        )}
      </div>
    </div>
  );
}

export default ReportPanel;
