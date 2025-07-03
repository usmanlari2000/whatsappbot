from sentence_transformers import SentenceTransformer
import json

data = [
    """Provident Fund (PF) Enrollment
Topic: PF Enrollment Process 
• When activated: Your PF status is activated in HRIS.
• Next steps (automated email): You will receive an email from HRIS (check your inbox and spam folder).
• Action required by employee: Complete the form, upload the affidavit, and fill the survey by the specified deadline in the email.
• Benefit: Ensures your PF contributions are processed smoothly.
• Contact for help: HR Operations team.""",

    """Health Insurance Claims (Non-Panel Hospitals)
Topic: Health Insurance Claims for Non-Panel Hospitals 
Provider: EFU Insurance Company.
• Policy: PITB's corporate health insurance (Master Policy ID: EM/001055-00).
• Scenario: For visits to hospitals NOT on EFU's approved panel list.
• How to submit: Claims must be submitted along with all original supporting receipts and the duly attested doctor's form.
• Required documents:
  o Completed EFU Health Insurance Claim Form.
  o Original Medical Bills/Invoices.
  o Original Prescriptions.
  o Diagnostic Reports (if applicable).
  o Discharge Summary (for hospitalization).
  o Copy of Employee's CNIC.
  o Copy of Patient's CNIC/B-Form (if dependent).
  o Doctor's Recommendation/Prescription.
• Contact for help: HR Wing for claim form and submission guidance.""",

    """HR Service Delivery Overview
Topic: HR Services Overview / How HR Can Help You 
Self-Service Portal: Access personal information, leave requests, payslips, benefits, and company policies.
• HR Contact: For complex queries or issues not resolved via self-service."""
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(data, normalize_embeddings=True)

output_data = [{"text": data[i], "embedding": embeddings[i].tolist()} for i in range(len(data))]

output_path = "minilm_embeddings.json"

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)
