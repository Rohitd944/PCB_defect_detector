from collections import Counter
from datetime import datetime

# A single high-confidence defect is enough to reject the board — it should
# not be hidden behind the "few detections" manual-inspection bucket.
FAIL_CONFIDENCE_PERCENT = 50.0


def generate_analysis(detections, filename):
    inspection_id = "PCB-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    total_defects = len(detections)

    if total_defects == 0:
        highest_confidence = 0
        average_confidence = 0

    else:

        confidences = [d["confidence"] for d in detections]

        highest_confidence = round(max(confidences) * 100, 2)

        average_confidence = round(
            sum(confidences) / total_defects * 100,
            2
        )
    defect_types = [d["type"] for d in detections]
    distribution = dict(Counter(defect_types))

    high_confidence_defect = highest_confidence > FAIL_CONFIDENCE_PERCENT

    if total_defects == 0:
        inspection_status = "PASS"

    elif high_confidence_defect or total_defects > 10:
        inspection_status = "FAIL"

    else:
        inspection_status = "MANUAL INSPECTION"

    if inspection_status == "PASS":

        recommendation = (
        "PCB passed inspection. "
        "No manufacturing defects detected."
    )

    elif inspection_status == "FAIL" and high_confidence_defect:

        recommendation = (
        f"A high-confidence defect ({highest_confidence}%) was detected. "
        "Reject the PCB and review the manufacturing process."
    )

    elif inspection_status == "FAIL":

        recommendation = (
        "PCB contains multiple defects. "
        "Reject the PCB and review the manufacturing process."
    )

    else:

        recommendation = (
        "PCB contains a moderate number of defects. "
        "Manual inspection is recommended."
    )
    observations = []
    observations.append(
    f"Total defects detected: {total_defects}"
)
    observations.append(
    f"Highest confidence: {highest_confidence}%"
)
    observations.append(
    f"Average confidence: {average_confidence}%"
)
    if distribution:
        most_common = max(
            distribution,
            key=distribution.get
        )
        observations.append(
            f"Most common defect: {most_common}"
        )
    else:
        observations.append("No defects detected")
    is_defective = total_defects > 0
    verdict = "DEFECTIVE" if is_defective else "DEFECT-FREE (PASSED)"

    return {
    "inspection_id": inspection_id,
    "timestamp": timestamp,
    "filename": filename,
    "total_defects": total_defects,
    "is_defective": is_defective,
    "verdict": verdict,
    "highest_confidence": highest_confidence,
    "average_confidence": average_confidence,
    "distribution": distribution,
    "inspection_status": inspection_status,
    "recommendation": recommendation,
    "observations": observations
}