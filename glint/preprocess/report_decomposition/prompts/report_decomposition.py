# ruff: noqa: E501
REPORT_DECOMPOSITION_PROMPT = """
You are an expert medical assistant AI specializing in understanding and analyzing chest radiology reports.

Your task is to extract all medically significant findings from the given report, including both abnormal and normal observations.

Follow these rules strictly:

1. Current Exam focus only:
  - Exclude any direct or implied references to prior studies.
  - Remove or rewrite temporal comparison terms such as "new", "improved", "unchanged", "worsened", "consistent".
  - Convert resolved/absent findings: When a finding is described as "resolved", "cleared", "absent", "no longer present", or similar terms indicating absence, convert it to a negative finding statement.
  - Convert all findings to present-tense declarative statements describing only the current exam.
  - Examples:
    * "Improved right pleural effusion." -> "There is right pleural effusion."
    * "The pleural effusion is resolved." -> "There is no pleural effusion."
    * "No longer present opacity." -> "There is no opacity."

2. Sentence format:
  - Each finding must be written as a single, complete declarative sentence.
  - If an anatomical location is provided, include it.
  - If no location is mentioned, describe the finding without inferring or adding a location.
  - Preserve uncertainty when present: maintain phrases like "may be", "could be", "suggestive of", "likely", "possibly" if they appear in the original.
  - Normal and negative findings (absence of finding) must also be reported as declarative sentences.
  - Do not omit any clinical finding, including normal anatomical structures and negative findings that is explicitly mentioned in the report.
  - Maintain medical accuracy: preserve the original medical terminology and meaning.

  - **Multiple findings separation:**
    * When a sentence contains multiple findings connected by commas, "and", "or", "with associated", "with", or other conjunctions, extract EACH finding as a separate sentence.
    * Example: "Bilateral pleural effusions, cardiomegaly and mild edema suggest fluid overload."
      → Extract: "There are bilateral pleural effusions.", "There is cardiomegaly.", "There is mild pulmonary edema.", "There is fluid overload."
    * Even if findings are part of a larger diagnostic statement, extract each component finding individually.

  - **Compound findings with descriptive context:**
    * When a SINGLE finding has additional descriptive or suggestive information attached to it, extract BOTH:
      * The complete finding with its descriptive context: "There is no consolidation to suggest pneumonia."
      * The core finding without the descriptive context: "There is no consolidation."
    * Apply this rule when findings are connected by phrases like "to suggest", "consistent with", "compatible with", "indicative of", "suggestive of", "likely due to", "possibly due to"
    * Note: This rule applies to individual findings after multiple findings have been separated.

3. Output format:
  - Return the result strictly as one valid JSON object.
  - Key: "report_decomposition"
  - Value: must be a list of extracted findings as strings.
  - Ensure the JSON is valid and parsable.
  - Example:
      {
        "report_decomposition": [
          "There is right pleural effusion.",
          "There is no pneumothorax.",
          "The cardiac silhouette is normal in size."
        ]
      }

4. Empty Array
  - Return {"report_decomposition": []} ONLY when the report contains no extractable clinical findings at all (e.g., "No interval change", "No comparison available", or completely empty/non-informative reports).
  - If the report contains ANY normal findings, negative findings, or clinical observations, you MUST include them in the output.
  - Examples of what should NOT return empty array:
    * "Normal heart size" → Include: "The heart size is normal."
    * "No pneumothorax" → Include: "There is no pneumothorax."
    * "Clear lungs" → Include: "The lungs are clear."
    * "No acute cardiopulmonary process." → Include: "There is no acute cardiopulmonary process."
  - Examples of what SHOULD return empty array:
    * "No interval change" (with no other findings describing current status)
    * "No comparison available" (with no other findings)
    * Empty or non-informative reports

Do not provide explanations, comments, or any text outside the JSON object. Output only the JSON object.

"""


CXR_REPORT_DECOMPOSITION_EXAMPLE = """
[EXAMPLE]
INPUT:
Cardiomegaly is accompanied by improving pulmonary vascular congestion and decreasing pulmonary edema. Left retrocardiac opacity has substantially improved, likely a combination of atelectasis and effusion. A more confluent opacity at the right lung base persists, and could be due to asymmetrically resolving edema, but pneumonia should be considered in the appropriate clinical setting. Small right pleural effusion is likely unchanged, with pigtail pleural catheter remaining in place and no visible pneumothorax. Bilateral pleural effusions, cardiomegaly and mild edema suggest fluid overload, however, given the clinical history, underlying consolidation due to pneumonia cannot be excluded at the lung bases.

OUTPUT:
{
  "report_decomposition" : [
    "There is cardiomegaly.",
    "There is pulmonary vascular congestion.",
    "There is pulmonary edema.",
    "There is left retrocardiac opacity.",
    "There may be atelectasis.",
    "There may be effusion.",
    "There is right lung base opacity.",
    "There is right lung base opacity suggestive of possible pneumonia.",
    "There may be small right pleural effusion.",
    "There is pigtail pleural catheter in place.",
    "There is no pneumothorax.",
    "There are bilateral pleural effusions.",
    "There is cardiomegaly.",
    "There is mild pulmonary edema.",
    "There is fluid overload.",
    "There may be consolidation at the lung bases.",
    "There may be consolidation due to pneumonia at the lung bases."
  ]
}
"""

CT_REPORT_DECOMPOSITION_EXAMPLE = """
[Example]
INPUT:
"Central venous catheter is seen on the right. The thoracic vertebral corpus heights, alignments and densities are normal. The catheter terminates in the right atrium. Heart contour and size are normal. There are atheromatous plaques in the aorta and coronary arteries. The widths of the mediastinal main vascular structures are normal. Pericardial effusion was not detected. There are lymph nodes in the mediastinum and hilar regions. The largest of these lymph nodes is observed in the subcarinal region and its short diameter is 15 mm. There is bilateral pleural effusion. The pleural effusion measured 50 mm on the right at its thickest point. The pleural effusion continues to the apex of both lungs when the patient is in the supine position. There is no pathological wall thickness increase in the esophagus within the sections. There is an occlusive hiatal hernia at the lower end of the esophagus. There is no obstructive pathology in the trachea and both main bronchi. There are uniform interlobular septal thickenings in both lungs. It was also observed in millimetric centriacinar nodules. It is understood that these findings are new. When evaluated together with the pleural effusion and the patient's clinical information, it was thought that the described manifestations might be due to pulmonary edema. It is recommended to evaluate the patient together with clinical and physical examination findings. Apart from these, there are small consolidations in the right lung upper lobe posterior segment and lower lobe superior segment. These appearances may be due to pulmonary edema. This appearance may be less likely in pneumonic infiltrates. It is recommended to evaluate the patient together with clinical and laboratory findings. Both lungs have millimetric nodules, some of which are calcific. No mass was detected in both lungs. No upper abdominal free fluid-collection was detected in the sections. There are no fractures or lytic-destructive lesions in the bone structures within the sections. Chronic renal failure in follow-up.  Bilateral pleural effusion, interlobular septal thickenings and centriacinar nodules in both lungs (patient is recommended to be evaluated for pulmonary edema).  Minor consolidations in the right lung, which may again be compatible with pulmonary edema or pneumonic infiltration.  Millimetric nodules in both lungs.  Mediastinal and hilar lymph nodes.  Atherosclerotic changes in the aorta and coronary arteries. Hiatal hernia."

OUTPUT:
{
  "report_decomposition": [
    "There is a central venous catheter of the right side.",
    "Thoracic vertebral corpus heights are normal.",
    "Thoracic vertebral alignments are normal.",
    "Thoracic vertebral densities are normal.",
    "There is a central venous catheter terminating in the right atrium.",
    "The heart contour is normal.",
    "The heart size is normal.",
    "There are atheromatous plaques of the aorta.",
    "There are atheromatous plaques of the coronary arteries.",
    "There are lymph nodes of the mediastinum.",
    "There are lymph nodes of the hilar regions.",
    "There is a subcarinal lymph node with a short diameter of 15 mm.",
    "There is bilateral pleural effusion.",
    "There is pleural effusion measuring 50 mm at the thickest point on the right.",
    "There is pleural effusion extending to the apex of both lungs.",
    "There is an occlusive hiatal hernia of the lower esophagus.",
    "There are uniform interlobular septal thickenings of both lungs.",
    "There are millimetric centriacinar nodules of both lungs.",
    "There are millimetric nodules of both lungs.",
    "There are calcific millimetric nodules of both lungs.",
    "There are small consolidations of the right upper lobe posterior segment.",
    "There are small consolidations of the right lower lobe superior segment.",
    "There may be pulmonary edema.",
    "There is no mediastinal widening.",
    "There is no pericardial effusion.",
    "There is no pathological esophageal wall thickening.",
    "There is no obstructive pathology of the trachea and main bronchi.",
    "There is no mass in either lung.",
    "There may be pneumonic infiltration.",
    "There is no upper abdominal free fluid collection.",
    "There are no fractures of the bone structures.",
    "There are no lytic-destructive lesions of the bone structures.",
    "There is a hiatal hernia."
  ]
}
"""
