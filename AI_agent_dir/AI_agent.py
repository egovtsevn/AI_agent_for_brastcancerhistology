import numpy as np
from tqdm import tqdm
from PIL import Image
import queue
import torch
from itertools import product
from openslide import OpenSlide
import threading
import json
import re
import time
from groq import Groq
from Bio import Entrez


class AI_agent:

    CAP_BREAST_BIOPSY_TEMPLATE = [
        "Procedure",
        "Specimen Laterality",
        "Tumor Site",
        "Histologic Type",
        "Histologic Grade (Nottingham)",
        "Tumor Size",
        "Percent Invasive Carcinoma",
        "Percent Non-Invasive Epithelium (unclassified)",
        "Percent Necrosis",
        "Lymphovascular Invasion",
        "Microcalcifications",
        "Margins",
        "Biomarkers (ER/PR/HER2/Ki-67)",
        "Additional Pathologic Findings",
        "Comment",
    ]

    FIELDS_FROM_IMAGE_ANALYSIS = {
        "Percent Invasive Carcinoma": "invasive epithelium",
        "Percent Non-Invasive Epithelium (unclassified)": "non-invasive epithelium",
        "Percent Necrosis": "necrosis",
    }

    FIELDS_REQUIRING_MANUAL_INPUT = [
        "Procedure", "Specimen Laterality", "Tumor Site", "Histologic Type",
        "Histologic Grade (Nottingham)", "Tumor Size", "Lymphovascular Invasion",
        "Microcalcifications", "Margins", "Biomarkers (ER/PR/HER2/Ki-67)",
        "Additional Pathologic Findings",
    ]

    def __init__(self,
                 base_model,
                 transform,
                 classifier,
                 groq_api_key: str = "...",
                 entrez_email: str = "...",
                 entrez_api_key: str = None):

        """ base_model: model that extracts  patch features
            classifier: model that classifies patches"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = base_model.to(self.device)
        self.transform = transform
        self.classifier = classifier
        self.patch_queue = queue.Queue(maxsize=32)
        self.result_dict = {}

        self.llm_client = Groq(api_key=groq_api_key)
        Entrez.email = entrez_email
        if entrez_api_key:
            Entrez.api_key = entrez_api_key

        self.pubmed_tool_schema = {
            "type": "function",
            "function": {
                "name": "search_pubmed",
                "description": (
                    "Searches PubMed for scientific articles. Use MeSH terms combined with "
                    "Title/Abstract terms for best precision. Combine tumor type with "
                    "relevant pathology terms (e.g. necrosis, tumor grading, prognosis)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "PubMed search query using standard syntax, e.g. "
                                '\'"osteosarcoma"[Title/Abstract] AND Necrosis[MeSH]\''
                            )
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of articles to return",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def GPU_processing(self):
        while True:
            item = self.patch_queue.get()
            if item is None:
                self.patch_queue.task_done()
                break
            idx, batch = item
            try:
                with torch.inference_mode():
                    tensor_batch = torch.stack(batch).to(self.device, non_blocking=True)
                    embeddings = self.base_model(tensor_batch)
                    preds = self.classifier.predict(embeddings.cpu().numpy())
                self.result_dict[idx] = preds
            except Exception as e:
                self.worker_error = e
                self.worker_error_idx = idx
            finally:
                self.patch_queue.task_done()

    def predict(self, tif, coords, dims,
                step=56, patch_size=224, batch_size=64):
        self.result_dict = {}
        self.worker_error = None
        worker = threading.Thread(target=self.GPU_processing)
        worker.start()

        W, H = dims[0], dims[1]
        base_W = ((W // step) * step) + patch_size
        base_H = ((H // step) * step) + patch_size
        base = np.full((base_H, base_W, 3), 0, dtype=np.uint8)

        img = OpenSlide(tif)
        region = img.read_region(coords, 0, dims).convert("RGB")
        img.close()
        region_np = np.array(region)
        base[0:H, 0:W] = region_np

        batch, batch_idx = [], 0
        i_range = list(range(0, H, step))
        j_range = list(range(0, W, step))
        total = len(i_range) * len(j_range)

        for i, j in tqdm(product(i_range, j_range), total=total, desc="Building patches"):
            patch = Image.fromarray(base[i:i + patch_size, j:j + patch_size])
            tensor = self.transform(patch)
            batch.append(tensor)
            if len(batch) == batch_size:
                self.patch_queue.put((batch_idx, batch))
                batch_idx += 1
                batch = []
        if batch:
            self.patch_queue.put((batch_idx, batch))

        self.patch_queue.put(None)
        worker.join()

        if self.worker_error is not None:
            raise RuntimeError(f"GPU_processing failed on batch {self.worker_error_idx}") from self.worker_error

        ordered_results = []
        for k in sorted(self.result_dict.keys()):
            ordered_results.extend(self.result_dict[k])

        expected_total = len(i_range) * len(j_range)
        if len(ordered_results) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} predictions, got {len(ordered_results)}."
            )

        return ordered_results, region_np, H, W, base_W, base_H

    def aggregate_predictions(self, tif, coords, dims, class_colors, class_names,
                              annotation=None, step=56, patch_size=224, batch_size=64):
        """
        Hard-voting aggregation among crossing pixels
        """
        prediction, region_np, H, W, base_W, base_H = self.predict(
            tif, coords, dims, step, patch_size, batch_size
        )

        num_classes = len(class_colors)
        n_steps_i = len(range(0, dims[1], step))
        n_steps_j = len(range(0, dims[0], step))

        votes = np.zeros((base_H, base_W, num_classes), dtype=np.int32)

        for i in range(n_steps_i):
            for j in range(n_steps_j):
                y, x = i * step, j * step
                label = i * n_steps_j + j
                pred_class = int(prediction[label])
                votes[y:y + patch_size, x:x + patch_size, pred_class] += 1

        final_mask = np.argmax(votes, axis=-1)
        final_mask_cropped = final_mask[:H, :W]

        total_pixels = final_mask_cropped.size
        fractions = {}
        for label, name in class_names.items():
            area_px = int(np.sum(final_mask_cropped == label))
            fractions[name] = {
                "area_px": area_px,
                "percent": round(100 * area_px / total_pixels, 2)
            }

        return {
            "fractions": fractions,
            "final_mask": final_mask_cropped,
            "region_np": region_np,
            "H": H, "W": W,
        }

    def analyze_large_slide_chunked(self,
                                 tif: str,
                                 coords: tuple[int, int],
                                 dims: tuple[int, int],
                                 class_colors: dict,
                                 class_names: dict,
                                 chunk_size: int = 4096,
                                 step: int = 56,
                                 patch_size: int = 224,
                                 batch_size: int = 64,
                                 annotation=None) -> dict:
      """
      Processes chosen slide area by chunks using pixel-level hard voting (aggregate_predictions)
      coords — (x, y) starting point on the slide (level 0).
      dims   — (W, H) soze of analyzed area
      """
      start_x, start_y = coords
      W_total, H_total = dims

      num_classes = len(class_colors)
      total_area_px = {name: 0 for name in class_names.values()}
      grand_total_pixels = 0

      chunk_coords = list(product(
          range(0, W_total, chunk_size),
          range(0, H_total, chunk_size)
      ))

      for cx, cy in tqdm(chunk_coords, desc="Processing chunks"):
          cw = min(chunk_size, W_total - cx)
          ch = min(chunk_size, H_total - cy)
          abs_x = start_x + cx
          abs_y = start_y + cy
          try:
              agg = self.aggregate_predictions(
                  tif=tif,
                  coords=(abs_x, abs_y),
                  dims=(cw, ch),
                  class_colors=class_colors,
                  class_names=class_names,
                  annotation=annotation,
                  step=step,
                  patch_size=patch_size,
                  batch_size=batch_size,
              )
          except RuntimeError as e:
              print(f"Chunk at ({abs_x},{abs_y}) size ({cw}x{ch}) failed: {e}")
              continue
          for name, stats in agg["fractions"].items():
              total_area_px[name] += stats["area_px"]

          grand_total_pixels += agg["final_mask"].size
          del agg

      fractions = {
          name: {
              "area_px": area,
              "percent": round(100 * area / grand_total_pixels, 2) if grand_total_pixels else 0.0
          }
          for name, area in total_area_px.items()
      }

      return {
          "fractions": fractions,
          "total_area_px": grand_total_pixels,
          "region_coords": coords,
          "region_dims": dims,
      }

    def search_pubmed(self, query: str, max_results: int = 3) -> dict:
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            pmids = record["IdList"]

            if not pmids:
                return {"query": query, "results": [], "note": "No articles found for this query."}

            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            articles = []
            for article in records["PubmedArticle"]:
                try:
                    title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                    abstract_parts = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
                    abstract = " ".join(str(p) for p in abstract_parts)
                    year = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "n/a")
                    pmid = str(article["MedlineCitation"]["PMID"])
                    articles.append({
                        "pmid": pmid, "title": title,
                        "abstract": abstract, "year": year
                    })
                except (KeyError, IndexError):
                    continue

            time.sleep(0.35)
            return {"query": query, "results": articles}

        except Exception as e:
            return {"query": query, "results": [], "error": str(e)}
        

    def verify_citations(self, text: str, tool_results: list) -> bool:
      """
      Checks if all of PMIDs mentioned in the response are retrived from PubMED
      """

      available_pmids = set()

      for result in tool_results:
          for article in result.get("results", []):
              pmid = article.get("pmid")
              if pmid:
                  available_pmids.add(str(pmid))

      cited_pmids = set(
          re.findall(r'\bPMID[:\s]*(\d+)\b', text, flags=re.IGNORECASE)
      )

      if not cited_pmids:
          return False

      return cited_pmids.issubset(available_pmids)


    def _format_manual_fields(self, manual_fields: dict) -> str:
      if not manual_fields:
          return "No additional clinical/procedural information was provided."

      lines = []
      for field, value in manual_fields.items():
          if value and value != "Not provided — requires manual/clinical input":
              lines.append(f"- {field}: {value}")

      if not lines:
          return "No additional clinical/procedural information was provided."

      return "Additional clinical/procedural information provided by the pathologist:\n" + "\n".join(lines)

    def run_literature_agent(self, fractions: dict, tumor_type: str,
                          is_representative: bool, manual_fields: dict = None,
                          max_searches: int = 3, model: str = "openai/gpt-oss-120b") -> dict:

        meta_note = (
            "Note: this analysis covers the FULL representative specimen section."
            if is_representative else
            "IMPORTANT CAVEAT: this analysis covers only a PARTIAL, arbitrarily selected "
            "region of the slide, not the full specimen. Do not treat the percentages as "
            "definitive clinical measurements — phrase any literature search and summary "
            "accordingly (general context, not precise prognostic claims tied to the exact percentage)."
        )

        system_prompt = (
            "You are a pathology literature research assistant supporting a College of American "
            "Pathologists (CAP) synoptic report for a breast core needle biopsy. "
            "You are given quantitative tissue composition data derived from histological image analysis "
            "of the biopsy sample. Search PubMed for relevant scientific context. "
            "Use the search_pubmed tool up to 3 times. Start with one precise query and only perform "
            "additional searches if the retrieved literature is insufficient. Use a MeSH-based query "
            "and broaden it if no results are found. Always include the tumor type in the query. "
            "Do not fabricate citations — only reference articles actually returned by the tool. "
            f"{meta_note} "

            "IMPORTANT — classifier limitations: "
            "The image classifier used to generate these percentages distinguishes 4 broad morphologic "
            "categories: 'invasive epithelium', 'non-invasive epithelium', 'necrosis', and 'other'. "

            "'Invasive epithelium' identifies tissue regions showing an infiltrative growth pattern "
            "consistent with invasive carcinoma, based on the morphology used to train the classifier. "
            "This is an automated morphologic classification, not a substitute for confirmatory "
            "immunohistochemical or molecular workup. The histologic diagnosis itself is already "
            "established (see the confirmed histologic type, if provided below) — report the invasive "
            "epithelium percentage as a tissue composition measurement, not as a separate diagnostic "
            "determination. "

            "'Non-invasive epithelium' identifies epithelial tissue with a non-infiltrative (contained) "
            "growth pattern. This category is heterogeneous and includes both benign/normal ductal or "
            "lobular epithelium and in-situ neoplastic epithelium (e.g., DCIS), which the classifier does "
            "not distinguish from one another. Do NOT interpret the percentage of 'non-invasive epithelium' "
            "as equivalent to a DCIS diagnosis or DCIS extent — it may represent normal tissue, benign "
            "proliferative changes, in-situ carcinoma, or a mixture of these. "

            "The 'other' category is a heterogeneous catch-all that may include stroma, adipose tissue, "
            "inflammatory infiltrate, erythrocytes/blood, and calcifications, among other structures. "
            "Do NOT interpret the percentage of 'other' as representing any single tissue type, and do "
            "not assign it prognostic or diagnostic meaning. "

            "IMPORTANT — output format constraint: "
            "Your final output is NOT a general summary. It must be written as the 'Comment' field "
            "of a CAP synoptic report for a breast core needle biopsy — concise, factual, structured "
            "as short evidence-linked statements, not narrative prose. "

            "This is a core needle biopsy, not a resection specimen. Do not comment on margins, "
            "pathologic tumor size (pT), or lymph node status — these are not assessable on core biopsy "
            "and are outside the scope of this report. "

            "Treat the image-derived tissue percentages strictly as measurements of tissue composition "
            "in the analyzed region of the biopsy core(s). Do not equate the percentage of invasive "
            "epithelium with tumor cellularity, tumor grade, tumor purity, or any other established "
            "clinical or histopathologic parameter unless the retrieved literature explicitly "
            "demonstrates that the same measurement is being used. "

            "Do not infer tumor grade, prognosis, stage, or other clinical characteristics directly "
            "from the measured tissue percentages. Clearly distinguish between: "
            "(1) findings directly demonstrated by the retrieved literature, "
            "(2) observations from the analyzed image, and "
            "(3) interpretations or hypotheses — label hypotheses explicitly as such. "

            "Do not describe a tissue component as a prognostic, predictive, or diagnostic factor "
            "unless the retrieved PubMed articles provide direct evidence for that association in "
            "the specified tumor type. "

            "When the analyzed region is partial (not the full specimen), explicitly state that the "
            "measurements represent only the sampled/imaged region and should not be extrapolated to "
            "the entire biopsy or the patient's overall disease burden."

            "You may be given additional clinical/procedural fields (e.g., histologic type, "
            "biomarker status, lymphovascular invasion). Use these only to: "
            "(1) refine the PubMed search query (e.g., include biomarker status or histologic "
            "subtype if relevant to finding more specific literature), and "
            "(2) avoid stating anything in your summary that contradicts these already-established "
            "findings. "
            "Do NOT combine these clinical fields with the image-derived tissue percentages to "
            "produce a new prognostic or diagnostic conclusion that is not directly supported by "
            "the retrieved literature. Each already-confirmed clinical finding should be treated as "
            "a given fact, not as something to re-derive or reinterpret.")

        clinical_context = self._format_manual_fields(manual_fields)

        user_prompt = (f"Tumor type: {tumor_type}\n"
                       f"{clinical_context}\n\n"
                       f"Tissue composition (from image analysis): {json.dumps(fractions, indent=2)}\n\n"
                       "Find relevant literature context for the significant tissue components "
                       "and prepare a short evidence-based summary "
                       "suitable for the Comment section of a CAP synoptic report. "
                       "End with a list of cited PMIDs.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        all_tool_results = []

        max_searches = max_searches

        for search_idx in range(max_searches):
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[self.pubmed_tool_schema],
                tool_choice="required" if search_idx == 0 else "auto",
            )

            msg = response.choices[0].message

            print(f"\n=== Search {search_idx} ===")
            print("finish_reason:", response.choices[0].finish_reason)
            print("content:", msg.content)
            print("tool_calls:", msg.tool_calls)

            assistant_message = {
                "role": "assistant",
                "content": msg.content,
            }

            if msg.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            messages.append(assistant_message)

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:

                args = json.loads(tool_call.function.arguments)

                result = self.search_pubmed(
                    query=args["query"],
                    max_results=min(args.get("max_results", 3), 3)
                )

                all_tool_results.append(result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })

        messages.append({
          "role": "user",
          "content": (
              "The literature search is now complete. "
              "Do not call any more tools. "
              "Using only the retrieved PubMed results, provide the final "
              "evidence-based summary suitable for the Comment section of a "
              "CAP-style pathology report. "
              "Cite only PMIDs that were actually returned by search_pubmed."
          )
        })

        response = self.llm_client.chat.completions.create(
          model=model,
          messages=messages,
          tools=[self.pubmed_tool_schema],
          tool_choice="none")

        msg = response.choices[0].message

        if not msg.content:
            return {
                "comment_text": "The literature agent failed to produce a final summary.",
                "citations_verified": False,
                "tool_results": all_tool_results,
            }

        verified = self.verify_citations(msg.content, all_tool_results)

        return {
            "comment_text": msg.content,
            "citations_verified": verified,
            "tool_results": all_tool_results,
        }

    def build_cap_synoptic_report(self, fractions: dict, manual_fields: dict = None) -> dict:
        manual_fields = manual_fields or {}
        report = {}

        for field in self.CAP_BREAST_BIOPSY_TEMPLATE:
            if field in self.FIELDS_FROM_IMAGE_ANALYSIS:
                key = self.FIELDS_FROM_IMAGE_ANALYSIS[field]
                stats = fractions.get(key)
                report[field] = f"{stats['percent']}%" if stats else "Not assessed"
            elif field in self.FIELDS_REQUIRING_MANUAL_INPUT:
                report[field] = manual_fields.get(field, "Not provided — requires manual/clinical input")
            else:
                report[field] = None

        return report

    def format_cap_report_text(self, cap_report: dict) -> str:
        lines = ["=" * 60, "CAP SYNOPTIC REPORT — BREAST CORE NEEDLE BIOPSY", "=" * 60]
        for field in self.CAP_BREAST_BIOPSY_TEMPLATE:
            value = cap_report.get(field, "Not provided")
            lines.append(f"\n{field}:")
            lines.append(f"  {value}")
        lines.append("\n" + "=" * 60)
        lines.append("This report was generated with AI-assisted image analysis and literature "
                      "retrieval. It requires review and verification by a qualified pathologist "
                      "before clinical use.")
        return "\n".join(lines)

    def generate_full_report(self, tif, coords, dims, class_colors, class_names,
                              tumor_type: str, manual_fields: dict = None,
                              is_representative: bool = False,
                              annotation=None, step=56, patch_size=224, batch_size=64) -> dict:

        if is_representative:
            agg = self.analyze_large_slide_chunked(tif,
                                                   coords,
                                                   dims,
                                                   class_colors,
                                                   class_names)
        else:
            agg = self.aggregate_predictions(
                tif, coords, dims, class_colors, class_names,
                annotation, step, patch_size, batch_size
            )
        fractions = agg["fractions"]
        self.last_fractions = fractions

        cap_report = self.build_cap_synoptic_report(fractions, manual_fields=manual_fields)
        try:
            literature = self.run_literature_agent(fractions,
                                                   tumor_type, is_representative,
                                                   manual_fields=manual_fields)
            cap_report["Comment"] = literature["comment_text"]
            citations_verified = literature["citations_verified"]
        except Exception as e:
            cap_report["Comment"] = f"[Literature agent unavailable: {e}]"
            citations_verified = False

        return {
            "cap_synoptic_report": cap_report,
            "raw_fractions": fractions,
            "region_representative": is_representative,
            "citations_verified": citations_verified,
        }