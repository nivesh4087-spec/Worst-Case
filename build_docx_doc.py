"""
DOCX Document Generator for RAG Best-Case vs Worst-Case Study
Generates `docs/Worst_Case_Study_HAT_RAG.docx`.
"""

import os
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def build_docx():
    doc = docx.Document()
    
    # Page setup
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = p_title.add_run("Comprehensive Worst-Case Performance Study & Algorithmic Bounds of Hierarchical Abstract Tree (HAT) in Cross-Document RAG")
    run_t.font.size = Pt(18)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

    # Metadata
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.add_run("Author: Nivesh Jain (Roll No: 44) | Co-Authors: Kaushalya Thopate, Rachit Ingole, Harsh Nayak, Hasan Rupawalla\n").bold = True
    p_meta.add_run("Department of Computer Engineering, Vishwakarma Institute of Technology, Pune, India\n")
    p_meta.add_run("Academic Year: 2026-2027 | Repository: https://github.com/nivesh4087-spec/Worst-Case")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Heading 1
    h1 = doc.add_heading("1. Executive Summary & Problem Formulation", level=1)
    doc.add_paragraph(
        "The Hierarchical Abstract Tree (HAT) architecture introduced by Jain et al. achieves exponential retrieval speedups "
        "O(B log N) and superior cross-document recall under balanced semantic clustering. However, real-world multi-document "
        "corpora frequently violate assumptions of semantic uniformity. This paper presents a detailed study comparing HAT's "
        "Best-Case performance against severe Worst-Case edge scenarios: Semantic Abstraction Drift, Fallback Cascading, "
        "and Degenerate Density Skew."
    )

    # Complexity Table
    h2 = doc.add_heading("2. Theoretical Complexity Matrix", level=1)
    table = doc.add_table(rows=5, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Algorithm", "Best-Case Time", "Worst-Case Time", "Space Complexity", "Worst-Case Quality"]
    
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    rows_data = [
        ["HAT (Proposed)", "O(B K log N)", "O(B log N + N d)", "O(N d)", "Recall -> 0.20 (Drift)"],
        ["Flat DPR", "O(N d)", "O(N d)", "O(N d)", "Recall -> 1.00 (Exact)"],
        ["RAPTOR", "O(B K log N)", "O(B K log N)", "O(N d L)", "Recall -> 0.00 (Pruned)"],
        ["HNSW Graph", "O(log N)", "O(N d)", "O(N M)", "Recall -> 0.60 (Local Min)"]
    ]
    
    for r_idx, r_data in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(r_data):
            row_cells[c_idx].text = val

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 3: Visual Benchmarks
    doc.add_heading("3. Empirical Visual Benchmarks", level=1)
    
    plots = [
        ("Figure 1: Latency Scaling vs Corpus Size (N)", "plots/1_latency_vs_dataset_size.png"),
        ("Figure 2: Recall@10 Comparison Across Datasets", "plots/2_recall_ndcg_comparison.png"),
        ("Figure 3: Vector Operations / FLOP Overhead", "plots/3_operation_count_flops.png"),
        ("Figure 4: Dynamic Fallback Search Trigger Rate (%)", "plots/4_tree_depth_degradation.png"),
        ("Figure 5: Multi-Dimensional Tradeoff Radar", "plots/5_tradeoff_radar.png")
    ]
    
    for cap, img_path in plots:
        if os.path.exists(img_path):
            doc.add_paragraph(cap).runs[0].font.bold = True
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.add_run().add_picture(img_path, width=Inches(5.5))
            doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Section 4: Architectural Recommendations
    doc.add_heading("4. Mitigation Strategies & Recommendations", level=1)
    doc.add_paragraph("1. Adaptive Soft Thresholding: Implement dynamic quantile routing instead of fixed tau cutoff.")
    doc.add_paragraph("2. Residual Vector Indexing: Store residual difference vectors to eliminate semantic abstraction drift.")
    doc.add_paragraph("3. Multi-beam Redundancy: Maintain candidate subtrees during low-confidence transitions.")

    os.makedirs("docs", exist_ok=True)
    out_path = "docs/Worst_Case_Study_HAT_RAG.docx"
    doc.save(out_path)
    print(f"Generated DOCX report successfully at `{out_path}`!")

if __name__ == "__main__":
    build_docx()
